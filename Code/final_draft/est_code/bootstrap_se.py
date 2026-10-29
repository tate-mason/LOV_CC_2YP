"""Household bootstrap for the two-stage demand estimator."""
import numpy as np
import pandas as pd

from price_control import add_price_controls


def resample_households(frame, rng):
    """Keep whole histories; repeated households get distinct bootstrap IDs.

    Stocks/Xi stay attached to the original histories. Rebuilding stocks after
    dropping incomplete price weeks would incorrectly erase observed purchases.
    """
    groups = [group for _, group in frame.groupby("household_code", sort=True)]
    selected = rng.integers(0, len(groups), size=len(groups))
    samples = []
    for new_id, old_id in enumerate(selected):
        group = groups[old_id].copy()
        group["household_code"] = new_id
        samples.append(group)
    return pd.concat(samples, ignore_index=True)


def bootstrap_standard_errors(frame, args, estimate, initial, names):
    """Refit price controls AND demand in every household-cluster replication.

    estimate(frame, initial) returns a scipy optimizer result. All replications
    use the same integration draws as the main fit. Retail instruments stay
    fixed: this is inference conditional on the observed retail price universe.
    """
    rng = np.random.default_rng(args.seed + 10000)
    records = []
    values = []
    for replication in range(args.bootstrap_reps):
        record = {"replication": replication + 1, "converged": False}
        try:
            sample = resample_households(frame, rng)
            if not args.no_price_control:
                sample, _ = add_price_controls(sample)
            fit = estimate(sample, initial)
            record.update(converged=bool(fit.success), objective=float(fit.fun),
                          message=str(fit.message))
            record.update(dict(zip(names, fit.x)))
            if fit.success and np.isfinite(fit.x).all() and np.isfinite(fit.fun):
                values.append(fit.x)
        except (ValueError, np.linalg.LinAlgError) as error:
            # An unidentifiable resample is recorded, never silently accepted.
            record["message"] = str(error)
        records.append(record)
        # Checkpoint after each replication in case a long cluster job ends.
        pd.DataFrame(records).to_csv(args.output / "bootstrap_draws.csv", index=False)
        print(f"Bootstrap {replication+1}/{args.bootstrap_reps}: "
              f"{len(values)} usable fits", flush=True)
    valid = len(values)
    report = dict(method=("household bootstrap" if args.no_price_control
                          else "household bootstrap, refitting both stages"),
                  requested=args.bootstrap_reps, usable=valid,
                  conditional_on="retail instruments and fixed integration draws")
    # A large failure fraction can select the bootstrap distribution.
    if valid < max(20, int(np.ceil(.8 * args.bootstrap_reps))):
        report["status"] = "insufficient converged replications; no SEs reported"
        return None, report
    estimates = np.asarray(values)
    table = pd.DataFrame({"parameter": names,
                          "std_error": estimates.std(axis=0, ddof=1),
                          "ci_lower": np.quantile(estimates, .025, axis=0),
                          "ci_upper": np.quantile(estimates, .975, axis=0)})
    covariance = pd.DataFrame(np.cov(estimates, rowvar=False), index=names, columns=names)
    covariance.to_csv(args.output / "covariance.csv")
    report["status"] = "computed; percentile 95% intervals"
    return table, report
