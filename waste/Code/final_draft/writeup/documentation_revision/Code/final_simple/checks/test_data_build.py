"""Generated in-memory/Parquet checks for the complete data build; no CSV reads."""
from pathlib import Path
from types import SimpleNamespace
import contextlib
import io
import tempfile
import unittest
from unittest.mock import patch

import polars as pl
from data import build_data as build


def panel_tables():
    purchases = pl.DataFrame({'trip_code_uc': [1, 2, 99], 'upc': [101, 102, 101],
                              'quantity': [2, 1, 1], 'total_price_paid': [2., 1., 1.], 'deal_flag_uc': [0, 0, 0]})
    trips = pl.DataFrame({'trip_code_uc': [1, 2], 'household_code': [10, 10],
                          'purchase_date': ['2010-01-02', '2010-01-09'],
                          'store_code_uc': [7, 7], 'panel_year': [2010, 2010]})
    panelists = pl.DataFrame({'household_code': [10], 'panel_year': [2010], 'household_size': [1]})
    products = pl.DataFrame({'upc': [101, 102], 'brand_descr': ['A', 'B']})
    return purchases.lazy(), trips.lazy(), panelists.lazy(), products.lazy()


def retail_tables():
    movements = pl.DataFrame({'upc': [101, 102], 'store_code_uc': [7, 7],
                              'week_end': [20100102, 20100109], 'price': [1., 2.]})
    rms = pl.DataFrame({'upc': [101, 101, 102], 'upc_ver_uc': [1, 2, 1]})
    products = pl.DataFrame({'upc': [101, 102], 'upc_ver_uc': [1, 1],
                             'product_module_code': [3603, 3612], 'brand_code_uc': [5, 6],
                             'brand_descr': ['A', 'B']})
    stores = pl.DataFrame({'store_code_uc': [7], 'year': [2010], 'retailer_code': [9]})
    extra = pl.DataFrame({'upc': [101, 101, 102], 'panel_year': [2010]*3,
                          'flavor_descr': [None, 'strawberry', 'plain']})
    brands = pl.DataFrame({'brand_code_uc': [5, 6], 'brand_name': ['A', 'B']})
    return tuple(table.lazy() for table in [movements, rms, products, stores, extra, brands])


class DataBuildChecks(unittest.TestCase):
    def test_tsv_conversion_options_without_reading_text(self):
        # Substitute a generated table for the raw reader; never open the TSV.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'input.tsv'
            source.touch()
            generated = pl.DataFrame({'UPC': [1], 'PRICE': [2.]}).lazy()
            with patch.object(pl, 'scan_csv', return_value=generated) as reader:
                build.convert_one_file(source, root / 'output.parquet', lossy_text=True)
            self.assertEqual(reader.call_args.kwargs['separator'], '\t')
            self.assertEqual(reader.call_args.kwargs['quote_char'], None)
            self.assertEqual(reader.call_args.kwargs['encoding'], 'utf8-lossy')
            self.assertEqual(pl.read_parquet(root / 'output.parquet').columns, ['upc', 'price'])

    def test_cleaning_matches_long_names_and_preserves_originals(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pl.DataFrame({'upc': [1], 'upc_ver_uc': [1], 'panel_year': [2010],
                          'flavor_code': ['a'], 'flavor_descr': ['berry'], 'unused': [99]}).write_parquet(root / 'products_extra_2010.parquet')
            build.clean_parquet_files(root, build.RETAIL_COLUMNS)
            cleaned = pl.read_parquet(root / 'cleaned_products_extra_2010.parquet')
            self.assertEqual(cleaned.columns, build.RETAIL_COLUMNS['products_extra'])
            self.assertIn('unused', pl.read_parquet(root / 'products_extra_2010.parquet').columns)
            build.clean_parquet_files(root, build.RETAIL_COLUMNS)
            self.assertFalse((root / 'cleaned_cleaned_products_extra_2010.parquet').exists())

    def test_panel_inner_trip_join_and_left_demographics(self):
        tables = panel_tables()
        merged = build.join_panel_tables(*tables).collect().sort('trip_code_uc')
        self.assertEqual(merged.height, 2)  # Unmatched trip 99 is discarded.
        self.assertEqual(merged['household_code'].to_list(), [10, 10])
        absent = pl.DataFrame({'household_code': [20], 'panel_year': [2010], 'household_size': [1]}).lazy()
        missing = build.join_panel_tables(tables[0], tables[1], absent, tables[3]).collect()
        self.assertEqual(missing.height, 2)
        self.assertEqual(missing['household_size'].null_count(), 2)

    def test_saturday_calendar_including_sunday_and_new_year(self):
        frame = pl.DataFrame({'purchase_date': ['2020-12-26', '2020-12-27', '2020-12-31', '2021-01-01', '2021-01-02']}).lazy()
        actual = build.add_saturday_week(frame).collect()['week_end'].to_list()
        self.assertEqual(actual, [20201226, 20210102, 20210102, 20210102, 20210102])
        # Independently retain the original formula for an equivalence check.
        date = pl.col('purchase_date').str.to_date()
        original = frame.with_columns((date + pl.duration(days=6-(date.dt.weekday()%7))).dt.strftime('%Y%m%d').cast(pl.Int64).alias('week_end'))
        self.assertEqual(actual, original.collect()['week_end'].to_list())

    def test_retail_first_version_nonmissing_flavor_and_left_joins(self):
        merged = build.join_retail_tables(*retail_tables()).collect().sort('upc')
        self.assertEqual(merged.height, 2)
        self.assertEqual(merged['upc_ver_uc'].to_list(), [1, 1])
        self.assertEqual(merged['flavor_descr'].to_list(), ['strawberry', 'plain'])
        self.assertEqual(merged['panel_year'].to_list(), [2010, 2010])

    def test_year_type_alignment_and_combination(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pl.DataFrame({'upc': ['1'], 'panel_year': [2010], 'flavor_code': [5]}).write_parquet(root / 'cleaned_products_extra_2010.parquet')
            pl.DataFrame({'upc': [2], 'panel_year': [2011], 'flavor_code': ['6']}).write_parquet(root / 'cleaned_products_extra_2011.parquet')
            with contextlib.redirect_stdout(io.StringIO()):
                build.combine_retail_files(root)
            table = pl.read_parquet(root / 'combined/products_extra.parquet').sort('upc')
            self.assertEqual(table['upc'].to_list(), [1, 2])
            self.assertEqual(table['flavor_code'].to_list(), ['5', '6'])
            self.assertEqual(table.schema['panel_year'], pl.Int32)

    def test_module_filter_and_exact_year_week_upc_store_matching(self):
        panel = build.add_saturday_week(build.join_panel_tables(*panel_tables()))
        retail = build.join_retail_tables(*retail_tables())
        quotes = build.yogurt_retail_quotes(retail, [3603])
        matched = build.match_one_year(panel, quotes, 2010).collect()
        self.assertEqual(matched['upc'].to_list(), [101])
        self.assertEqual(build.match_one_year(panel, quotes, 2011).collect().height, 0)

    def test_file_merge_week_and_match_pipeline(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            combined = root / 'combined'
            combined.mkdir()
            for table, name in zip(panel_tables(), ['purchases', 'trips', 'panelists', 'products']):
                destination = root / 'cleaned_products.parquet' if name == 'products' else combined / f'{name}.parquet'
                table.sink_parquet(destination)
            retail_dir = root / 'RMS'
            (retail_dir / 'combined').mkdir(parents=True)
            movements, rms, products, stores, extra, brands = retail_tables()
            movements.filter(pl.col('upc')==101).sink_parquet(retail_dir / 'combined/3603.parquet')
            movements.filter(pl.col('upc')==102).sink_parquet(retail_dir / 'combined/3612.parquet')
            rms.sink_parquet(retail_dir / 'combined/rms.parquet')
            products.sink_parquet(retail_dir / 'cleaned_products.parquet')
            stores.sink_parquet(retail_dir / 'combined/stores.parquet')
            extra.sink_parquet(retail_dir / 'combined/products_extra.parquet')
            brands.sink_parquet(retail_dir / 'brand_variations.parquet')
            settings = SimpleNamespace(data_dir=root, first_year=2010, last_year=2010,
                                       run_steps=['merge_panel', 'panel_weeks', 'merge_retail', 'match_yogurt'],
                                       retail_merged_input=None, yogurt_module_codes=[3603,3612])
            build.run_build(settings)
            final = pl.read_parquet(root / 'yogurt_estimation_sample.parquet').sort('upc')
            self.assertEqual(final.height, 2)
            self.assertEqual(final['price'].to_list(), [1.,2.])
            self.assertEqual(final['week_end'].to_list(), [20100102,20100109])
            self.assertEqual(final['household_code'].to_list(), [10,10])
            self.assertTrue((retail_dir / 'merged_rms.parquet').exists())
            self.assertEqual(list(root.glob('yogurt-years-*')), [])


if __name__ == '__main__':
    unittest.main()
