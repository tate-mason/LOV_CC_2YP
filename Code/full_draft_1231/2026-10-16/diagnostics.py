"""Design diagnostics: numerical separation is different from causal identification."""
import numpy as np
import pandas as pd


def design_diagnostics(data, output):
    rows, households = [], []
    j = len(data.flavors)
    for p in data.panels:
        local = []
        z = np.log1p(p.stock)
        for t in np.flatnonzero(p.keep):
            offered = np.flatnonzero(p.available[t])
            if len(offered) < 2:
                continue
            reference = offered[0]
            for k in offered[1:]:
                intercept = np.eye(j)[k, 1:] - np.eye(j)[reference, 1:]
                row = np.r_[intercept, p.prices[t, k] - p.prices[t, reference],
                            -(z[t, k] - z[t, reference])]
                rows.append(row)
                local.append((reference, k, row[-2], row[-1]))
        if local:
            frame = pd.DataFrame(local, columns=['reference', 'alternative', 'price', 'history'])
            # Center separately by household AND flavor pair; menus can change.
            centered = frame[['price', 'history']] - frame.groupby(
                ['reference', 'alternative'])[['price', 'history']].transform('mean')
            households.append({'household': p.household,
                               'purchase_weeks': len(np.unique(p.weeks)),
                               'flavors_bought': int((p.counts.sum(axis=0) > 0).sum()),
                               'within_pair_history_sd': float(centered.history.std(ddof=0)),
                               'within_pair_price_sd': float(centered.price.std(ddof=0))})
    x = np.asarray(rows)
    if not len(x):
        raise ValueError('No rows with two available alternatives.')
    norms = np.linalg.norm(x, axis=0)
    singular = np.linalg.svd(x / np.where(norms > 0, norms, 1), compute_uv=False)
    # How much history remains after projecting on flavor constants and price?
    residual = x[:, -1] - x[:, :-1] @ np.linalg.lstsq(x[:, :-1], x[:, -1], rcond=None)[0]
    pd.DataFrame(households).to_csv(output / 'household_variation.csv', index=False)
    return {'parameters_in_homogeneous_design': int(x.shape[1]),
            'relative_rows': len(x), 'rank': int(np.linalg.matrix_rank(x)),
            'column_scaled_singular_values': singular.tolist(),
            'history_residual_rms': float(np.sqrt(np.mean(residual**2))),
            'history_rms': float(np.sqrt(np.mean(x[:, -1]**2))),
            'interpretation': 'Full rank separates homogeneous log-odds coefficients under maintained assumptions; it does not prove mixed-logit identification or history/price exogeneity.'}
