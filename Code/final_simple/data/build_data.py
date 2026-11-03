"""Build the purchase data, from raw TSV files through matched yogurt purchases.

Edit Parameters, select the steps you want, then run python3 data/build_data.py.
Functions below follow the same order as the research workflow. Importing this
file does not read data, create output folders, or start any conversion.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse
import os
import tempfile

import polars as pl


# === Parameters: edit these paths and choose the steps to run === #

# These are the existing server locations. Change them for another machine.
data_dir = Path('/scratch/dtm63837/Kilts_Panel/Data')
panel_raw = Path('/scratch/dtm63837/Kilts_Panel/nielsen_extracts/nielsen_extracts/nielsen_extracts/HMS')
retail_raw = Path('/scratch/dtm63837/Kilts_Panel/nielsen_extracts/nielsen_extracts/RMS')
first_year = 2006
last_year = 2015
convert_annual = True
convert_master = True

# Choose explicitly; an empty list prints instructions and does not access data.
# For a complete rebuild, copy the full list of steps below into run_steps.
run_steps = []
# run_steps = [
#     'convert_panel', 'clean_panel', 'combine_panel', 'merge_panel', 'panel_weeks',
#     'convert_retail', 'clean_retail', 'combine_retail', 'merge_retail', 'match_yogurt',
# ]

# Default output from merge_retail, also the input for match_yogurt.
# The old scripts disagreed on this path. Both stages now use Data/RMS/merged_rms.parquet.
retail_merged_input = None  # Set a Path here only if using an existing file elsewhere.
yogurt_module_codes = [3603, 3612]

# The dictionaries list the raw columns kept for downstream calculations.
# Changing these lists changes what information is available after cleaning.
PANEL_COLUMNS = {
    'panelists': [pl.col('household_cd').alias('household_code'), 'panel_year',
                  'household_income', 'household_size', 'male_head_age', 'female_head_age',
                  'male_head_employment', 'female_head_employment', 'race', 'hispanic_origin',
                  'dma_cd', 'dma_name'],
    'purchases': ['trip_code_uc', 'upc', 'quantity', 'total_price_paid', 'deal_flag_uc'],
    'trips': ['trip_code_uc', 'household_code', 'purchase_date', 'store_code_uc', 'panel_year'],
    'products': ['upc', 'product_module_code', 'product_module_descr', 'product_group_code',
                 'product_group_descr', 'brand_code_uc', 'brand_descr', 'size1_amount', 'size1_units'],
}
RETAIL_COLUMNS = {
    '3603': ['store_code_uc', 'upc', 'week_end', 'price'],
    '3612': ['store_code_uc', 'upc', 'week_end', 'price'],
    'products': ['upc', 'upc_ver_uc', 'product_module_code', 'brand_code_uc', 'brand_descr',
                 'size1_code_uc', 'size1_amount', 'size1_units'],
    'products_extra': ['upc', 'upc_ver_uc', 'panel_year', 'flavor_code', 'flavor_descr'],
    'stores': ['store_code_uc', 'year', 'retailer_code', 'channel_code', 'dma_code', 'dma_descr'],
}
RETAIL_TYPES = {'flavor_code': pl.String, 'size1_code_uc': pl.String,
                'upc': pl.Int64, 'store_code_uc': pl.Int64, 'panel_year': pl.Int32}
MATCH_KEYS = ['panel_year', 'week_end', 'upc', 'store_code_uc']
AVAILABLE_STEPS = [
    'convert_panel', 'clean_panel', 'combine_panel', 'merge_panel', 'panel_weeks',
    'convert_retail', 'clean_retail', 'combine_retail', 'merge_retail', 'match_yogurt',
]


# === 1. Convert raw text files and keep the required columns === #


def convert_one_file(input_path, output_path, lossy_text=False):
    """Read a tab-separated raw file, lowercase its column names, write Parquet.

    A missing file is reported and skipped, matching the old conversion loops.
    Read those messages to check coverage; skipping does not establish completeness.
    """
    if not input_path.exists():
        print(f'Missing raw file; skipped: {input_path}', flush=True)
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    encoding = 'utf8-lossy' if lossy_text else 'utf8'
    raw = pl.scan_csv(input_path, separator='\t', quote_char=None,
                      infer_schema_length=10000, encoding=encoding)
    raw.rename(str.lower).sink_parquet(output_path)
    print(f'Converted {input_path.name} -> {output_path}', flush=True)


def convert_panel_files(raw_dir, output_dir, years, annual=True, master=True):
    if annual:
        for year in years:
            for table in ['panelists', 'purchases', 'trips']:
                input_path = raw_dir / str(year) / 'Annual_Files' / f'{table}_{year}.tsv'
                convert_one_file(input_path, output_dir / f'{table}_{year}.parquet')
    if master:
        for table in ['products', 'retailers']:
            input_path = raw_dir / 'Master_Files' / 'Latest' / f'{table}.tsv'
            convert_one_file(input_path, output_dir / f'{table}.parquet', lossy_text=True)


def convert_retail_files(raw_dir, output_dir, years, annual=True, master=True):
    if annual:
        for year in years:
            # RMS version files are named rms_versions in raw data, rms in outputs.
            for raw_name, table in [('products_extra', 'products_extra'),
                                     ('rms_versions', 'rms'), ('stores', 'stores')]:
                input_path = raw_dir / str(year) / 'Annual_Files' / f'{raw_name}_{year}.tsv'
                convert_one_file(input_path, output_dir / f'{table}_{year}.parquet')
            for module in ['3603', '3612']:
                input_path = raw_dir / str(year) / 'Movement_Files' / f'2510_{year}' / f'{module}_{year}.tsv'
                convert_one_file(input_path, output_dir / f'{module}_{year}.parquet')
    if master:
        for table in ['brand_variations', 'products']:
            input_path = raw_dir / 'Master_Files_2006-2020' / 'Latest' / f'{table}.tsv'
            convert_one_file(input_path, output_dir / f'{table}.parquet', lossy_text=True)


def clean_parquet_files(directory, columns_by_table):
    """Write cleaned_ copies. Do not change or delete the original conversions."""
    # Test longer names first: products_extra must not be mistaken for products.
    table_names = sorted(columns_by_table, key=len, reverse=True)
    for input_path in sorted(directory.glob('*.parquet')):
        if input_path.name.startswith('cleaned_'):
            continue
        for table in table_names:
            if input_path.stem == table or input_path.stem.startswith(table + '_'):
                output_path = directory / ('cleaned_' + input_path.name)
                source = pl.scan_parquet(input_path)
                source.select(columns_by_table[table]).sink_parquet(output_path)
                print(f'Kept {table} columns: {output_path}', flush=True)
                break


# === 2. Combine annual files === #


def combine_panel_files(directory):
    combined = directory / 'combined'
    combined.mkdir(parents=True, exist_ok=True)
    for table in ['panelists', 'purchases', 'trips']:
        # This preserves the original wildcard rule: all matching years are stacked.
        files = sorted(directory.glob(f'cleaned_{table}_*.parquet'))
        if not files:
            raise FileNotFoundError(f'No cleaned annual panel files for {table}: {directory}')
        pl.scan_parquet(files).sink_parquet(combined / f'{table}.parquet')
        print(f'Combined {len(files)} files for {table}', flush=True)


def align_retail_types(frame):
    """Make identifiers/codes compatible across annual retail schemas."""
    names = frame.collect_schema().names()
    casts = []
    for column, dtype in RETAIL_TYPES.items():
        if column in names:
            casts.append(pl.col(column).cast(dtype))
    return frame.with_columns(casts)


def combine_retail_files(directory):
    combined = directory / 'combined'
    combined.mkdir(parents=True, exist_ok=True)
    patterns = {'stores': 'cleaned_stores_*.parquet',
                'products_extra': 'cleaned_products_extra_*.parquet',
                '3612': 'cleaned_3612_*.parquet', '3603': 'cleaned_3603_*.parquet',
                'rms': 'rms_*.parquet'}
    for table, pattern in patterns.items():
        files = sorted(directory.glob(pattern))
        if not files:
            print(f'No annual files for {table}; skipped {directory / pattern}', flush=True)
            continue
        annual_frames = []
        for path in files:
            annual_frames.append(align_retail_types(pl.scan_parquet(path)))
        pl.concat(annual_frames, how='vertical_relaxed').sink_parquet(combined / f'{table}.parquet')
        print(f'Combined {len(files)} files for {table}', flush=True)


# === 3. Merge purchases with household, trip, and product information === #


def join_panel_tables(purchases, trips, panelists, products):
    """Pure table calculation: purchases are the anchor, trips use an inner join.

    Demographic and product joins are left joins. No new deduplication or sample
    restriction is imposed here; the original join rules are retained.
    """
    purchases = purchases.with_columns(pl.col('trip_code_uc').cast(pl.Int64), pl.col('upc').cast(pl.Int64))
    trips = trips.with_columns(pl.col('trip_code_uc').cast(pl.Int64),
                                pl.col('household_code').cast(pl.Int64), pl.col('panel_year').cast(pl.Int32))
    panelists = panelists.with_columns(pl.col('household_code').cast(pl.Int64), pl.col('panel_year').cast(pl.Int32))
    products = products.with_columns(pl.col('upc').cast(pl.Int64))
    merged = purchases.join(trips, on='trip_code_uc', how='inner')
    merged = merged.join(panelists, on=['household_code', 'panel_year'], how='left')
    merged = merged.join(products, on='upc', how='left')
    return merged


def merge_panel_files(directory):
    combined = directory / 'combined'
    purchases = pl.scan_parquet(combined / 'purchases.parquet')
    trips = pl.scan_parquet(combined / 'trips.parquet')
    panelists = pl.scan_parquet(combined / 'panelists.parquet')
    products = pl.scan_parquet(directory / 'cleaned_products.parquet')
    merged = join_panel_tables(purchases, trips, panelists, products)
    output_path = directory / 'merged_panel.parquet'
    merged.sink_parquet(output_path)
    audit_file(output_path, anchor=purchases, keys=['trip_code_uc', 'household_code', 'upc', 'panel_year'])


def add_saturday_week(frame):
    """Attach YYYYMMDD Saturday week-end, including Sunday and year boundaries."""
    date = pl.col('purchase_date').cast(pl.String).str.to_date()
    saturday = date + pl.duration(days=(6 - date.dt.weekday()) % 7)
    return frame.with_columns(saturday.dt.strftime('%Y%m%d').cast(pl.Int64).alias('week_end'))


def attach_panel_weeks(directory):
    # A temporary replacement avoids reading/writing the same Parquet simultaneously.
    panel_path = directory / 'merged_panel.parquet'
    with tempfile.TemporaryDirectory(prefix='panel-week-', dir=directory) as work:
        replacement = Path(work) / 'merged_panel.parquet'
        add_saturday_week(pl.scan_parquet(panel_path)).sink_parquet(replacement)
        os.replace(replacement, panel_path)
    print(f'Attached Saturday week-end: {panel_path}', flush=True)


# === 4. Merge retail movements with store and product metadata === #


def join_retail_tables(movements, rms, products, stores, products_extra, brand_variations):
    """Retain the original key casts, first-row rules, and left-join sequence."""
    movements = movements.with_columns(
        pl.col('upc').cast(pl.Int64), pl.col('store_code_uc').cast(pl.Int64),
        (pl.col('week_end') // 10000).cast(pl.Int32).alias('panel_year'))
    rms = rms.with_columns(pl.col('upc').cast(pl.Int64), pl.col('upc_ver_uc').cast(pl.Int32))
    rms = rms.unique(subset=['upc'], keep='first')
    products = products.with_columns(pl.col('upc').cast(pl.Int64), pl.col('upc_ver_uc').cast(pl.Int32))
    products = products.unique(subset=['upc', 'upc_ver_uc'], keep='first')
    stores = stores.rename({'year': 'panel_year'})
    stores = stores.with_columns(pl.col('store_code_uc').cast(pl.Int64), pl.col('panel_year').cast(pl.Int32))
    stores = stores.unique(subset=['store_code_uc', 'panel_year'], keep='first')
    products_extra = products_extra.with_columns(pl.col('upc').cast(pl.Int64), pl.col('panel_year').cast(pl.Int32))
    products_extra = products_extra.sort('flavor_descr', descending=False, nulls_last=True)
    products_extra = products_extra.unique(subset=['upc', 'panel_year'], keep='first')
    brand_variations = brand_variations.with_columns(pl.col('brand_code_uc').cast(pl.Int64))
    brand_variations = brand_variations.unique(subset=['brand_code_uc'], keep='first')

    merged = movements.join(rms, on='upc', how='left', suffix='_rms')
    merged = merged.join(products, on=['upc', 'upc_ver_uc'], how='left', suffix='_prod')
    merged = merged.join(stores, on=['store_code_uc', 'panel_year'], how='left', suffix='_str')
    merged = merged.join(products_extra, on=['upc', 'panel_year'], how='left', suffix='_xtra')
    merged = merged.join(brand_variations, on='brand_code_uc', how='left', suffix='_brand')
    return merged


def merge_retail_files(directory):
    combined = directory / 'combined'
    movements = pl.concat([pl.scan_parquet(combined / '3612.parquet'),
                           pl.scan_parquet(combined / '3603.parquet')])
    merged = join_retail_tables(
        movements, pl.scan_parquet(combined / 'rms.parquet'),
        pl.scan_parquet(directory / 'cleaned_products.parquet'),
        pl.scan_parquet(combined / 'stores.parquet'),
        pl.scan_parquet(combined / 'products_extra.parquet'),
        pl.scan_parquet(directory / 'brand_variations.parquet'))
    output_path = directory / 'merged_rms.parquet'
    merged.sink_parquet(output_path)
    audit_file(output_path, anchor=movements, keys=['upc', 'store_code_uc', 'panel_year'],
               metadata=['brand_descr', 'retailer_code', 'flavor_descr', 'brand_code_uc'])


# === 5. Match yogurt purchases to retail quotes === #


def yogurt_retail_quotes(retail, module_codes):
    """One retained retail quote for each year/week/UPC/store matching key."""
    needed = MATCH_KEYS + ['product_module_code', 'price']
    return (retail.select(needed).filter(pl.col('product_module_code').is_in(module_codes))
            .unique(subset=MATCH_KEYS, keep='first'))


def match_one_year(panel, retail_quotes, year):
    """Only purchases with a matching quote survive this inner join."""
    panel_year = panel.filter(pl.col('panel_year') == year)
    retail_year = retail_quotes.filter(pl.col('panel_year') == year)
    return panel_year.join(retail_year, on=MATCH_KEYS, how='inner', suffix='_rms')


def match_yogurt_files(directory, retail_path, years, module_codes):
    # Retain the old memory strategy: materialize projected yogurt quotes once,
    # then stream purchases one year at a time. Temporary files are isolated so
    # leftover chunks from an earlier interrupted run cannot enter this output.
    retail = pl.scan_parquet(retail_path)
    retail_quotes = yogurt_retail_quotes(retail, module_codes).collect()
    panel = pl.scan_parquet(directory / 'merged_panel.parquet')
    output_path = directory / 'yogurt_estimation_sample.parquet'
    with tempfile.TemporaryDirectory(prefix='yogurt-years-', dir=directory) as work:
        year_files = []
        for year in years:
            year_file = Path(work) / f'yogurt_{year}.parquet'
            match_one_year(panel, retail_quotes.lazy(), year).sink_parquet(year_file)
            year_files.append(year_file)
            print(f'Matched purchase year {year}', flush=True)
        pl.scan_parquet(year_files).sink_parquet(output_path)
    audit_file(output_path, keys=MATCH_KEYS, show_years=True)


# === 6. Report counts, missing keys, metadata coverage, prices, and file size === #


def audit_file(path, anchor=None, keys=(), metadata=(), show_years=False):
    """Report what the file contains; a row-count ratio alone is not a join proof."""
    frame = pl.scan_parquet(path)
    names = frame.collect_schema().names()
    count = frame.select(pl.len()).collect().item()
    print(f'\nOutput: {path}\nRows: {count:,}\nColumns: {names}', flush=True)
    if count == 0:
        raise ValueError(f'Empty merged output; inspect inputs and matching keys: {path}')
    if anchor is not None:
        original_count = anchor.select(pl.len()).collect().item()
        ratio = 100 * count / original_count if original_count else float('nan')
        print(f'Anchor rows: {original_count:,}; output/anchor: {ratio:.2f}%', flush=True)
    for column in keys:
        if column in names:
            missing = frame.select(pl.col(column).null_count()).collect().item()
            print(f'Missing {column}: {missing:,} ({100*missing/count:.2f}%)', flush=True)
    for column in metadata:
        if column in names:
            covered = frame.select(pl.col(column).is_not_null().sum()).collect().item()
            print(f'Nonmissing {column}: {covered:,} ({100*covered/count:.2f}%)', flush=True)
    for column in ['household_code', 'hh_id', 'upc', 'store_code_uc']:
        if column in names:
            unique = frame.select(pl.col(column).n_unique()).collect().item()
            print(f'Distinct {column}: {unique:,}', flush=True)
    if show_years and 'panel_year' in names:
        print(frame.group_by('panel_year').len().sort('panel_year').collect(), flush=True)
    price_column = next((c for c in ['price', 'price_rms'] if c in names), None)
    if price_column:
        stats = frame.select(
            pl.col(price_column).min().alias('minimum'), pl.col(price_column).mean().alias('mean'),
            pl.col(price_column).median().alias('median'), pl.col(price_column).max().alias('maximum'),
            pl.col(price_column).null_count().alias('missing'),
            (pl.col(price_column) <= 0).sum().alias('zero_or_negative')).collect()
        print(f'Price summary ({price_column}):\n{stats}', flush=True)
    print(f'File size: {path.stat().st_size / 1024**2:.2f} MB', flush=True)


# === 7. Optional server flags; ordinary use is editing Parameters above === #


def read_command_line(settings):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path)
    parser.add_argument('--panel-raw', type=Path)
    parser.add_argument('--retail-raw', type=Path)
    parser.add_argument('--first-year', type=int)
    parser.add_argument('--last-year', type=int)
    parser.add_argument('--steps', nargs='+', choices=AVAILABLE_STEPS, dest='run_steps')
    parser.add_argument('--retail-merged-input', type=Path)
    parser.add_argument('--skip-annual', dest='convert_annual', action='store_false')
    parser.add_argument('--skip-master', dest='convert_master', action='store_false')
    parser.set_defaults(**vars(settings))
    return parser.parse_args()


# === 8. Run the selected build steps, in their dependency order === #


def run_build(settings):
    if not settings.run_steps:
        print('Choose run_steps in the Parameters section. Available steps:')
        print(', '.join(AVAILABLE_STEPS))
        return
    if settings.first_year > settings.last_year:
        raise ValueError('first_year must not exceed last_year')
    unknown = set(settings.run_steps) - set(AVAILABLE_STEPS)
    if unknown:
        raise ValueError(f'Unknown build steps: {sorted(unknown)}')
    directory = settings.data_dir
    retail_directory = directory / 'RMS'
    years = range(settings.first_year, settings.last_year + 1)
    directory.mkdir(parents=True, exist_ok=True)
    retail_directory.mkdir(parents=True, exist_ok=True)

    if 'convert_panel' in settings.run_steps:
        convert_panel_files(settings.panel_raw, directory, years, settings.convert_annual, settings.convert_master)
    if 'clean_panel' in settings.run_steps:
        clean_parquet_files(directory, PANEL_COLUMNS)
    if 'combine_panel' in settings.run_steps:
        combine_panel_files(directory)
    if 'merge_panel' in settings.run_steps:
        merge_panel_files(directory)
    if 'panel_weeks' in settings.run_steps:
        attach_panel_weeks(directory)
    if 'convert_retail' in settings.run_steps:
        convert_retail_files(settings.retail_raw, retail_directory, years, settings.convert_annual, settings.convert_master)
    if 'clean_retail' in settings.run_steps:
        clean_parquet_files(retail_directory, RETAIL_COLUMNS)
    if 'combine_retail' in settings.run_steps:
        combine_retail_files(retail_directory)
    if 'merge_retail' in settings.run_steps:
        merge_retail_files(retail_directory)
    if 'match_yogurt' in settings.run_steps:
        retail_path = settings.retail_merged_input or retail_directory / 'merged_rms.parquet'
        match_yogurt_files(directory, retail_path, years, settings.yogurt_module_codes)


if __name__ == '__main__':
    settings = SimpleNamespace(
        data_dir=data_dir, panel_raw=panel_raw, retail_raw=retail_raw,
        first_year=first_year, last_year=last_year, convert_annual=convert_annual,
        convert_master=convert_master, run_steps=run_steps,
        retail_merged_input=retail_merged_input, yogurt_module_codes=yogurt_module_codes)
    # Providing flags overrides only those values for this run.
    import sys
    if len(sys.argv) > 1:
        settings = read_command_line(settings)
    run_build(settings)
