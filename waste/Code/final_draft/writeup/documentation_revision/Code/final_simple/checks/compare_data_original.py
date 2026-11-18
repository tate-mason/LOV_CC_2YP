"""Compare data joins and full summary outputs with the original code.

Uses generated Parquet and in-memory export captures only. It never imports
old raw-build scripts (which execute at import) and never opens a CSV file.
"""
from pathlib import Path
import ast
import contextlib
import importlib.util
import io
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import polars as pl
from data import build_data, summaries
from checks.test_data_build import panel_tables, retail_tables
from checks.test_data_summaries import sample_inputs, run_sample, compare_exports


def original_join(script, variables, inputs):
    """Evaluate only original table assignments, substituting generated inputs.

    Imports, file writes, audit scans, and top-level execution are excluded.
    The actual casts/deduplication/join expressions remain the original AST.
    """
    class SubstituteInput(ast.NodeTransformer):
        def visit_Call(self,node):
            if isinstance(node.func,ast.Attribute) and node.func.attr=='scan_parquet':
                # Each original scan uses os.path.join(base_dir, 'relative/path').
                relative=node.args[0].args[-1].value
                return ast.copy_location(ast.Subscript(value=ast.Name(id='inputs',ctx=ast.Load()),
                                                       slice=ast.Constant(relative),ctx=ast.Load()),node)
            return self.generic_visit(node)
    selected=[]
    for node in ast.parse(script.read_text()).body:
        if isinstance(node,ast.Assign) and any(isinstance(target,ast.Name) and target.id in variables for target in node.targets):
            selected.append(SubstituteInput().visit(node))
    module=ast.fix_missing_locations(ast.Module(body=selected,type_ignores=[]))
    environment={'pl':pl,'inputs':inputs}
    exec(compile(module,str(script),'exec'),environment)
    return environment['merged'].collect()


def compare(original):
    purchases,trips,panelists,products=panel_tables()
    panel_inputs={'combined/purchases.parquet':purchases, 'combined/trips.parquet':trips,
                  'combined/panelists.parquet':panelists,'cleaned_products.parquet':products}
    old_panel=original_join(original/'data_code/cleaning_merging/panel_merge.py',
                            {'purchases','trips','panelists','products','merged'},panel_inputs)
    new_panel=build_data.join_panel_tables(purchases,trips,panelists,products).collect()
    assert old_panel.sort('trip_code_uc').equals(new_panel.sort('trip_code_uc'))
    movements,rms,products,stores,extra,brands=retail_tables()
    retail_inputs={'combined/3612.parquet':movements.filter(pl.col('upc')==102),
                   'combined/3603.parquet':movements.filter(pl.col('upc')==101),
                   'combined/rms.parquet':rms,'cleaned_products.parquet':products,
                   'combined/stores.parquet':stores,'combined/products_extra.parquet':extra,
                   'brand_variations.parquet':brands}
    old_retail=original_join(original/'data_code/cleaning_merging/retail_merge.py',
                             {'movements','rms','products','stores','products_extra','brand_var','merged'},retail_inputs)
    new_retail=build_data.join_retail_tables(movements,rms,products,stores,extra,brands).collect()
    assert old_retail.sort('upc').equals(new_retail.sort('upc'))
    print('Panel and retail casts, deduplication, columns and joins match original expressions.')

    spec=importlib.util.spec_from_file_location('original_summaries',original/'data_code/summ_stats.py')
    old=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory)
        _,reference=sample_inputs(root)
        for batches in [1,3]:
            old_out,old_tables=run_sample(old,root,reference,f'original_{batches}',batches)
            new_out,new_tables=run_sample(summaries,root,reference,f'simple_{batches}',batches)
            compare_exports(old_tables,new_tables)
            original_prepared=pl.read_parquet(old_out/'yogurt_estimation_sample.parquet').sort(['household_code','purchase_date','upc'])
            simple_prepared=pl.read_parquet(new_out/'yogurt_estimation_sample.parquet').sort(['household_code','purchase_date','upc'])
            assert original_prepared.equals(simple_prepared)
            print(f'Matched all {len(old_tables)} exported summary tables and the prepared sample with {batches} household partitions.')
    print('Data comparisons passed. No CSV files opened; legacy CSV intermediates were supplied from generated memory tables.')


if __name__=='__main__':
    original=Path(sys.argv[1]) if len(sys.argv)>1 else ROOT.parent/'final_draft'
    compare(original)
