"""Compare shared flavor tastes with persistent random tastes on identical data."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import demand_core as core
from diagnostics import design_diagnostics


def make_model(data, args, cap):
    model = core.Model(data, args.draws, args.seed)
    k = len(data.flavors) - 1
    model.bounds[k:2*k] = [(0, cap)] * k
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default=core.DEFAULT_DATA_PATH)
    parser.add_argument('--menu')
    parser.add_argument('--flavor-column', default='flavor')
    parser.add_argument('--top-flavors', type=int, default=6)
    parser.add_argument('--prefiltered', action='store_true')
    parser.add_argument('--delta', type=float, default=0.8)
    parser.add_argument('--draws', type=int, default=128)
    parser.add_argument('--starts', type=int, default=3)
    parser.add_argument('--maxiter', type=int, default=400)
    parser.add_argument('--seed', type=int, default=219)
    parser.add_argument('--sd-caps', type=float, nargs='+', default=[0, 4, 8])
    parser.add_argument('--profile', type=float, nargs='+', default=[-1, -0.25, 0, 0.1, 0.5, 1])
    parser.add_argument('--recovery-reps', type=int, default=0)
    parser.add_argument('--recovery-gammas', type=float, nargs='+', default=[0, 0.5])
    parser.add_argument('--output', default='results')
    parser.add_argument('--audit-only', action='store_true')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        core.self_test()
        return
    if not 0 < args.delta < 1 or min(args.draws, args.starts, args.maxiter) < 1 or args.top_flavors < 2:
        parser.error('Invalid decay, counts, or flavor count.')
    if args.recovery_reps < 0 or any(not np.isfinite(c) or c < 0 for c in args.sd_caps):
        parser.error('SD caps must be finite/nonnegative and recovery reps nonnegative.')
    if any(not np.isfinite(g) or abs(g) > 8 for g in args.profile + args.recovery_gammas):
        parser.error('Gamma values must be finite and within [-8,8].')
    args.attribute_columns, args.timing_controls, args.heterogeneity = [], False, 'random'
    output = core.local_path(args.output)
    # Prevent stale optional diagnostics from being interpreted as a new run.
    if output.exists() and any(output.iterdir()):
        parser.error('Output directory is nonempty; choose a fresh --output.')
    data, audit = core.prepare(args)
    output.mkdir(parents=True, exist_ok=True)
    audit['design'] = design_diagnostics(data, output)
    audit['arguments'] = vars(args)
    (output / 'audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    core.identification(data).to_csv(output / 'flavor_variation.csv', index=False)
    data.attrition.to_csv(output / 'household_attrition.csv', index=False)
    if args.audit_only:
        print(f'Audit saved to {output}')
        return
    comparison, recovered = [], []
    for cap in sorted(set([0.0] + args.sd_caps)):
        label = f'sd_cap_{cap:g}'
        folder = output / label
        folder.mkdir()
        model = make_model(data, args, cap)
        print(f'Fitting {label}', flush=True)
        fit = model.fit(args.starts, maxiter=args.maxiter, seed=args.seed)
        null = model.fit(args.starts, fit.x, maxiter=args.maxiter, no_fatigue=True)
        estimates = pd.DataFrame({'parameter': model.names, 'estimate': fit.x})
        estimates['fixed'] = [lo == hi for lo, hi in model.bounds]
        estimates['at_bound'] = [np.isclose(v, lo) or np.isclose(v, hi)
                                 for v, (lo, hi) in zip(fit.x, model.bounds)]
        estimates.to_csv(folder / 'estimates.csv', index=False)
        pd.DataFrame({'parameter': model.names, 'estimate': null.x}).to_csv(folder / 'no_history_estimates.csv', index=False)
        table = core.profile(model, fit, args.profile, args.starts, args.maxiter)
        table.to_csv(folder / 'gamma_profile.csv', index=False)
        summary = {'sd_cap': cap, 'nll': float(fit.fun), 'gamma': float(fit.x[model.gamma_start]),
                   'converged': bool(fit.success), 'optimizer_message': str(fit.message),
                   'start_objectives': fit.start_objectives, 'no_history_nll': float(null.fun),
                   'no_history_converged': bool(null.success),
                   'no_history_beats_full': bool(null.fun < fit.fun - 1e-5),
                   'profile_beats_full': bool(table.profile_beats_unrestricted.any()),
                   'free_parameter_at_bound': bool((estimates.at_bound & ~estimates.fixed).any()),
                   'actual_draws': model.r}
        comparison.append(summary)
        (folder / 'fit.json').write_text(json.dumps(summary, indent=2) + '\n')
        # Test explicit zero and positive gamma truths, not just fitted truth.
        for gamma in args.recovery_gammas:
            truth = fit.x.copy()
            truth[model.gamma_start] = gamma
            for repetition in range(args.recovery_reps):
                generated = core.simulate(model, truth, args.seed + 1000 + repetition)
                estimator = make_model(generated, args, cap)
                result = estimator.fit(args.starts, maxiter=args.maxiter, seed=args.seed + repetition)
                for name, value, estimate in zip(model.names, truth, result.x):
                    recovered.append({'sd_cap': cap, 'gamma_truth': gamma, 'replication': repetition,
                                      'parameter': name, 'truth': value, 'estimate': estimate,
                                      'error': estimate-value, 'converged': bool(result.success)})
                print(f'Recovery {label}, gamma={gamma}, replication={repetition+1}', flush=True)
        pd.DataFrame(comparison).to_csv(output / 'model_comparison.csv', index=False)
        if recovered:
            pd.DataFrame(recovered).to_csv(output / 'recovery.csv', index=False)
    print(pd.DataFrame(comparison).to_string(index=False))
    print(f'Results saved to {output}')


if __name__ == '__main__':
    main()
