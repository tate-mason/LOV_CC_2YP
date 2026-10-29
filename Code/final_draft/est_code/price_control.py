"""Other-DMA instruments and the price first stage. See README.md."""
import numpy as np
import pandas as pd

from scipy.stats import t as student_t

FLAVORS = ["other", "berry", "plain"]


def check_dma(values, description):
    """Normalize Nielsen's numeric DMA identifiers across panel and RMS files."""
    codes = pd.to_numeric(values, errors="coerce")
    valid = codes.notna() & (codes > 0) & (codes == np.floor(codes))
    if not valid.all():
        raise ValueError(f"Missing/invalid {description} DMA codes: {(~valid).sum():,}")
    return codes.astype("int64")


def other_dma_instruments(weekly, exposure_upcs, dma_prices):
    """Match each exposure-store UPC to its same-week price in OTHER DMAs.

    dma_prices has ONE mean package price per UPC/week/retail DMA. First average
    these means over all DMAs except the CONSUMER's DMA, then average over the
    exposure store's UPCs in a flavor. This matches the local price's UPC basket.
    """
    dma_prices = dma_prices.copy()
    dma_prices["dma_code"] = check_dma(dma_prices["dma_code"], "retail")
    pk = ["upc", "week"]
    totals = (dma_prices.groupby(pk)["dma_price"].agg(["sum", "count"])
              .rename(columns={"sum": "sum_dma", "count": "n_dma"}).reset_index())
    exposure = weekly[["store_code_uc", "week", "dma_cd"]].drop_duplicates()
    exposure = exposure.merge(exposure_upcs, on=["store_code_uc", "week"],
                              how="left", validate="many_to_many")
    exposure = exposure.merge(totals, on=pk, how="left", validate="many_to_one")
    own = dma_prices.rename(columns={"dma_code": "dma_cd", "dma_price": "own_dma_price"})
    exposure = exposure.merge(own, on=pk + ["dma_cd"], how="left", validate="many_to_one")
    present = exposure["own_dma_price"].notna()
    denominator = exposure["n_dma"] - present.astype(int)
    exposure["iv"] = ((exposure["sum_dma"] - exposure["own_dma_price"].fillna(0))
                      / denominator.where(denominator > 0))
    # Do not silently change the UPC basket when an instrument is missing.
    keys = ["store_code_uc", "week", "dma_cd", "flavor"]
    grouped = exposure.groupby(keys)["iv"].agg(["mean", "count", "size"])
    grouped.loc[grouped["count"] != grouped["size"], "mean"] = np.nan
    result = (grouped["mean"].unstack("flavor").reindex(columns=FLAVORS)
              .rename(columns={f: "iv_" + f for f in FLAVORS}).reset_index())
    return weekly.merge(result, on=["store_code_uc", "week", "dma_cd"], how="left",
                        validate="many_to_one")


def absorb_effects(values, groups, tolerance=1e-10, maxiter=2000):
    """Remove week and DMA means without a huge matrix of dummy variables.

    Repeated demeaning is needed because DMA/week cells are unbalanced.
    Each household-week exposure has equal weight in the first stage.
    """
    residual = np.asarray(values, dtype=float).copy()
    codes = [pd.factorize(group, sort=True)[0] for group in groups]
    for _ in range(maxiter):
        previous = residual.copy()
        for code in codes:
            counts = np.bincount(code)
            means = np.zeros((len(counts), residual.shape[1]))
            np.add.at(means, code, residual)
            means /= counts[:, None]
            residual -= means[code]
        if np.max(np.abs(residual - previous)) < tolerance:
            return residual
    raise ValueError("First-stage fixed-effect demeaning did not converge.")


def add_price_controls(frame):
    """OLS separately by flavor: price ~ IV + all three Xi + week FE + DMA FE.

    Refit this function in every household bootstrap sample. Return residuals
    in DOLLARS (no arbitrary rescaling) for the choice-model control function.
    """
    result = frame.copy()
    reports = []
    hh, households = pd.factorize(result["household_code"], sort=True)
    for flavor in FLAVORS:
        columns = ["price_" + flavor, "iv_" + flavor] + ["xi_" + f for f in FLAVORS]
        values = result[columns].to_numpy(float)
        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite first-stage inputs for {flavor}.")
        within = absorb_effects(values, [result["week"], result["dma_cd"]])
        y, z, controls = within[:, 0], within[:, 1], within[:, 2:]
        # Partial out Xi before estimating the excluded instrument's coefficient.
        # This tolerates redundant Xi columns, but NOT a redundant instrument.
        z_partial = z - controls @ np.linalg.lstsq(controls, z, rcond=None)[0]
        y_partial = y - controls @ np.linalg.lstsq(controls, y, rcond=None)[0]
        z_ss = z_partial @ z_partial
        if z_ss <= 1e-12 * max(1., values[:, 1] @ values[:, 1]):
            raise ValueError(f"No excluded-instrument variation after controls for {flavor}.")
        pi = (z_partial @ y_partial) / z_ss
        control_beta = np.linalg.lstsq(controls, y - pi*z, rcond=None)[0]
        residual = y - pi*z - controls @ control_beta
        if np.std(residual) < 1e-10:
            raise ValueError(f"First-stage residual for {flavor} is essentially zero.")
        result["cf_" + flavor] = residual
        # Household-clustered partial-IV t/F diagnostic (not a weak-IV-robust test).
        scores = np.bincount(hh, weights=z_partial * residual, minlength=len(households))
        pi_se = np.sqrt((scores @ scores) / z_ss**2)
        if len(households) > 1:
            pi_se *= np.sqrt(len(households) / (len(households) - 1))
        f_stat = float((pi / pi_se)**2) if pi_se > 0 else np.nan
        restricted_ss = y_partial @ y_partial
        partial_r2 = 1 - (residual @ residual) / restricted_ss if restricted_ss > 0 else np.nan
        reports.append(dict(flavor=flavor, iv_coefficient=float(pi),
                            iv_cluster_se=float(pi_se), iv_cluster_F=f_stat,
                            iv_pvalue=float(2*student_t.sf(np.sqrt(f_stat), len(households)-1))
                            if len(households) > 1 else np.nan,
                            partial_R2=float(partial_r2), residual_sd=float(np.std(residual)),
                            observations=len(result), households=len(households)))
    return result, pd.DataFrame(reports)
