"""Full summary checks with generated Parquet inputs and in-memory exports."""
from pathlib import Path
from types import SimpleNamespace
import contextlib
import io
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import polars as pl
from data import summaries


@contextlib.contextmanager
def capture_exports():
    """Capture would-be CSV outputs in memory; never read a CSV file.

    The reader substitutes are also used by the optional original-code comparison
    because the old summary code rereads its own generated CSVs internally.
    """
    saved = {}
    def pandas_export(frame, filename, *args, **kwargs):
        saved[str(Path(filename))] = frame.copy()
    def polars_export(frame, filename, *args, **kwargs):
        if isinstance(frame, pl.LazyFrame):
            frame = frame.collect(engine='streaming')
        saved[str(Path(filename))] = frame.to_pandas()
    def memory_scan(path, *args, **kwargs):
        path = Path(path)
        if path.suffix == '.parquet':
            return pl.scan_parquet(path)
        if str(path) not in saved:
            raise AssertionError(f'CSV access prohibited: {path}')
        return pl.from_pandas(saved[str(path)]).lazy()
    def memory_read(path, *args, **kwargs):
        if str(Path(path)) not in saved:
            raise AssertionError(f'CSV access prohibited: {path}')
        frame = saved[str(Path(path))].copy()
        if kwargs.get('usecols'):
            frame = frame[kwargs['usecols']]
        return frame
    with patch.object(pd.DataFrame, 'to_csv', pandas_export), \
         patch.object(pl.DataFrame, 'write_csv', polars_export), \
         patch.object(pl.LazyFrame, 'sink_csv', polars_export), \
         patch.object(pl, 'scan_csv', memory_scan), \
         patch.object(pd, 'read_csv', memory_read):
        yield saved


def sample_inputs(root):
    source = pl.DataFrame({
        'household_code': [1,1,1,1,2,2,3,4,4], 'panel_year': [2010]*9,
        'purchase_date': ['2010-01-02','2010-01-09','2010-01-09','2010-01-23','2010-01-02','2010-01-09','2010-01-02','2010-01-02','2010-01-09'],
        'trip_code_uc': [1,2,2,3,4,5,6,7,8], 'upc': [1,1,2,3,2,3,1,1,2],
        'quantity': [2,1,1,3,1,2,1,1,1], 'race': [1,1,1,1,2,2,3,1,1],
        'hispanic_origin': [2,2,2,2,1,1,2,2,2], 'household_income': [5,5,5,5,8,8,9,7,7],
        'female_head_age': [3,3,3,3,0,0,4,3,3], 'male_head_age': [4,4,4,4,2,2,5,4,4],
        'household_size': [1,1,1,1,1,1,1,2,2], 'store_code_uc': [100]*9,
        'price': [1.]*9, 'original_metadata': ['preserved']*9,
    })
    for name in ['merged_panel','yogurt_estimation_sample']:
        source.write_parquet(root / f'{name}.parquet')
    reference = root / 'reference.parquet'
    pl.DataFrame({'upc': ['001','002','003','004'],
                  'flavor_descr': ['vanilla','strawberry','plain','unused']}).write_parquet(reference)
    return source, reference


def run_sample(module, root, reference, name, batches, figures=False):
    out, work = root / name, root / (name+'_work')
    out.mkdir()
    work.mkdir()
    settings = SimpleNamespace(data_dir=root, flavors=reference,
                               products_extra=root/'absent.parquet', first_year=2006,
                               last_year=2015, household_batches=batches, no_figures=not figures)
    with capture_exports() as saved, contextlib.redirect_stdout(io.StringIO()):
        module.run(settings, out, work)
    return out, {str(Path(path).relative_to(out)): frame for path, frame in saved.items()}


def ordered(frame):
    if frame.index.name:
        return frame.sort_index()
    return frame.sort_values(list(frame.columns), na_position='last').reset_index(drop=True)


def compare_exports(left, right):
    if set(left) != set(right):
        raise AssertionError(f'Output names differ: {set(left)^set(right)}')
    for name in left:
        pd.testing.assert_frame_equal(ordered(left[name]), ordered(right[name]),
                                      check_dtype=False, check_exact=False, rtol=1e-12, atol=1e-12, obj=name)


class DataSummaryChecks(unittest.TestCase):
    def test_single_person_records_before_two_week_rule(self):
        source = pl.DataFrame({'household_code':[1,1,2,2,3,3],
                                'purchase_date':['2010-01-02','2010-01-09']*3,
                                'household_size':['1','2','1','1',None,'1']}).lazy()
        source = source.with_columns(pl.col('purchase_date').str.to_date())
        repeated = summaries.repeated_yogurt_households(summaries.single_person_purchases(source)).collect()
        self.assertEqual(repeated['household_code'].unique().to_list(),[2])

    def test_full_pipeline_batch_invariance_and_prepared_quote_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, reference = sample_inputs(root)
            out1, saved1 = run_sample(summaries, root, reference, 'one', 1)
            out3, saved3 = run_sample(summaries, root, reference, 'three', 3)
            compare_exports(saved1, saved3)
            prepared = pl.read_parquet(out3/'yogurt_estimation_sample.parquet')
            self.assertEqual(set(prepared['household_code']), {1,2,4})
            self.assertEqual(prepared.height,8)
            self.assertTrue(set(source.columns).issubset(prepared.columns))
            self.assertEqual(prepared['original_metadata'].unique().to_list(),['preserved'])
            household = saved3['yogurt_household_flavors.csv'].set_index('household_code')
            self.assertEqual(set(household.index),{1,2})  # Size-two household supplies prices only.
            self.assertAlmostEqual(household.loc[1,'flavor_HHI'],19/49)
            self.assertAlmostEqual(household.loc[2,'flavor_HHI'],5/9)
            metrics = saved3['yogurt_dynamics.csv'].set_index('metric')['value']
            self.assertEqual(metrics['Eligible adjacent single-flavor week pairs'],1)
            self.assertEqual(metrics['Switch rate in eligible pairs (%)'],100)
            self.assertEqual(saved3['yogurt_weekly_flavor_quantities.csv'].shape[0],6)
            self.assertTrue((out3/'yogurt_household_flavors.parquet').exists())
            self.assertTrue((out3/'run_notes.txt').exists())

    def test_price_filter_precedes_repeat_week_rule(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, reference = sample_inputs(root)
            unpriced = source.with_columns(pl.when(
                ((pl.col('household_code')==2)&(pl.col('purchase_date')=='2010-01-09'))
                | ((pl.col('household_code')==1)&(pl.col('purchase_date')=='2010-01-09')&(pl.col('upc')==1)))
                .then(None).otherwise(pl.col('price')).alias('price'))
            unpriced.write_parquet(root/'yogurt_estimation_sample.parquet')
            out, saved = run_sample(summaries, root, reference, 'filtered', 2)
            prepared = pl.read_parquet(out/'yogurt_estimation_sample.parquet')
            self.assertEqual(set(prepared['household_code']),{1,4})
            self.assertEqual(prepared.height,5)
            self.assertEqual(saved['yogurt_price_coverage.csv']['purchase_lines_without_price_coverage'].iloc[0],2)

    def test_recovery_preserves_existing_blank_and_earliest_available_label(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            pl.DataFrame({'upc':['001','002'],'flavor_descr':['vanilla',None]}).write_parquet(root/'ref.parquet')
            pl.DataFrame({'upc':[1,2,3,3,4], 'flavor_descr':['plain','berry','cherry','strawberry',None],
                          'panel_year':[2006,2006,2007,2006,2006]}).write_parquet(root/'extra.parquet')
            with capture_exports():
                summaries.recover_flavor_reference(root/'ref.parquet',root/'extra.parquet',root/'result.parquet')
                result = summaries.flavor_reference(root/'result.parquet',detailed=True).sort('upc')
            self.assertEqual(result.to_dict(as_series=False),
                             {'upc':['1','2','3'],'flavor':['vanilla','unknown','strawberry']})

    def test_hhi_and_other_figures_render_from_parquet(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            _,reference=sample_inputs(root)
            out,_=run_sample(summaries,root,reference,'figures',2,figures=True)
            for folder in [out,out/'aggregated_flavors']:
                self.assertTrue((folder/'flavor_hhi_histogram.pdf').exists())
                self.assertTrue((folder/'flavor_hhi_cdf.pdf').exists())
            with patch.object(pd,'read_csv',side_effect=AssertionError('No CSV reads')):
                summaries.make_hhi_figures(out)


if __name__=='__main__':
    unittest.main()
