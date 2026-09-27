import sys
import tempfile
from pathlib import Path
from datetime import date
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import data_merge as m
import polars as pl

def test():
    with tempfile.TemporaryDirectory() as d:
        m.HMS_PATH = str(Path(d) / 'hms.parquet')
        m.RMS_PATH = str(Path(d) / 'rms.parquet')
        m.OUTPUT_DIR = d
        m.YEARS = (2022, 2023)
        panel = pl.DataFrame(dict(
            household_code=[1]*3, trip_code_uc=[1,2,3], household_size=[1]*3,
            product_module_code_hms=['3603']*3, size1_amount_hms=[6000]*3,
            size1_unit_hms=['OZ']*3, quantity=[1]*3, deal_flag_uc=[0]*3,
            store_code_uc=[10,10,99], upc=[123]*3,
            purchase_date=['20220101','2022-01-02','2022-01-03 12:00:00'],
            household_income=[1]*3, male_head_age=[30]*3, female_head_age=[0]*3,
            flavor=['berry']*3, flavor_cd=[0]*3))
        panel.write_parquet(m.HMS_PATH)
        rms = pl.DataFrame(dict(week_end=[date(2022,1,1),date(2022,1,8),date(2022,1,8),date(2022,1,8)], store_code_uc=[10,10,10,77], upc=[123]*4, price=[2.,3.,3.,100.]))
        rms.write_parquet(m.RMS_PATH)
        result=m.main().collect()
        assert result.height == 3
        assert sorted(result['price'].drop_nulls().to_list()) == [2.,3.]
        old=Path(d,'scanner_panel_2022.parquet').read_bytes()
        rms.with_columns(pl.Series('price',[2.,3.,4.,100.])).write_parquet(m.RMS_PATH)
        try: m.main()
        except RuntimeError as e: assert 'conflicting' in str(e)
        else: raise AssertionError('conflict accepted')
        assert Path(d,'scanner_panel_2022.parquet').read_bytes() == old
        rms.with_columns(pl.lit(999).alias('store_code_uc')).write_parquet(m.RMS_PATH)
        try: m.main()
        except RuntimeError as e: assert 'no RMS matches' in str(e)
        else: raise AssertionError('zero matches accepted')
        for values in [['20220108'], ['2022-01-08'], ['2022-01-08 12:00:00'], [date(2022,1,8)]]:
            assert pl.DataFrame({'d':values}).select(m.parse_date('d')).item() == date(2022,1,8)
    print('PASS: synthetic merge, duplicates, conflicts, unmatched rows, zero matches, empty year, date formats, safe publication')
if __name__ == "__main__":
    test()
