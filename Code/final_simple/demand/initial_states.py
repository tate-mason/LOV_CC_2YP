"""Pure first-basket selection; callers keep empirical frames on the server."""


def first_observed_baskets(weekly, flavors, calendar=None):
    keys = ['household_code', 'week']
    if weekly.duplicated(keys).any():
        raise ValueError('Expected unique household/week baskets.')
    if calendar is not None:
        if calendar.duplicated(keys).any():
            raise ValueError('Expected unique observed calendar weeks.')
        purchase_weeks = calendar.loc[calendar.raw_yogurt > 0, keys]
    else:
        purchase_weeks = weekly.loc[weekly[flavors].sum(axis=1) > 0, keys]
    first = purchase_weeks.sort_values(keys).drop_duplicates('household_code', keep='first')
    result = first.merge(weekly[keys+flavors], on=keys, how='left', validate='one_to_one')
    result['state_observed'] = result[flavors].notna().all(axis=1)
    return result
