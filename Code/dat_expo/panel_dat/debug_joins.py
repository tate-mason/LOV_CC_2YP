import os
import polars as pl

output_dir = "/scratch/dtm63837/Kilts_Panel/nielsen_extracts/output_markets"
test_market = "Atlanta"  # Change this to any market you want to test

print("=" * 60)
print(f"RUNNING JOIN DIAGNOSTIC TRACER FOR: {test_market}")
print("=" * 60)

# ---------------------------------------------------------
# STEP 1: Inspect Base Market Panelists File
# ---------------------------------------------------------
panelist_path = os.path.join(output_dir, f"{test_market}_panelists.parquet")

if not os.path.exists(panelist_path):
    print(f"❌ CRITICAL ERROR: {panelist_path} does not exist!")
    print("   The market filtering loop failed to write an output file.")
    exit(1)

p_df = pl.scan_parquet(panelist_path).collect()
print(f"\n1. Market Panelists ({test_market}): {p_df.height:,} rows")

if p_df.height == 0:
    print(f"❌ FAILING AT STEP 1: {test_market}_panelists.parquet is EMPTY (0 rows).")
    print("   Cause: FIPS codes or household_size filter matched 0 rows in panel_merge.py.")
    exit(1)

# ---------------------------------------------------------
# STEP 2: Inspect Trips Data
# ---------------------------------------------------------
trips_path = os.path.join(output_dir, "trips.parquet")
t_df = pl.scan_parquet(trips_path).collect()
print(f"2. Total Raw Trips: {t_df.height:,} rows")

# Check Trip <-> Panelist Join
tp_df = t_df.join(p_df, on=["panel_year", "household_code"], how="inner")
print(f"3. Trips + Panelists Inner Join: {tp_df.height:,} rows")

if tp_df.height == 0:
    print("\n❌ FAILING AT STEP 3: Join between TRIPS and PANELISTS produced 0 rows.")
    print("   Data Type Diagnostics:")
    print("   - Panelist household_code dtype:", p_df["household_code"].dtype)
    print("   - Trips household_code dtype:   ", t_df["household_code"].dtype)
    print("   - Panelist panel_year dtype:     ", p_df["panel_year"].dtype)
    print("   - Trips panel_year dtype:        ", t_df["panel_year"].dtype)
    print("\n   Sample Values:")
    print("   - Sample Panelist HHs:", p_df["household_code"].head(5).to_list())
    print("   - Sample Trips HHs:   ", t_df["household_code"].head(5).to_list())
    print("   - Sample Panelist Years:", p_df["panel_year"].unique().to_list())
    print("   - Sample Trips Years:   ", t_df["panel_year"].unique().to_list())
    
    # Cross-check if HH overlap exists ignoring year
    hh_overlap = set(p_df["household_code"]).intersection(set(t_df["household_code"]))
    print(f"   - HH code overlap (ignoring panel_year): {len(hh_overlap)} matching IDs")
    exit(1)

# ---------------------------------------------------------
# STEP 3: Inspect Purchases Data
# ---------------------------------------------------------
purchases_path = os.path.join(output_dir, "purchases.parquet")
pur_df = pl.scan_parquet(purchases_path).collect()
print(f"4. Total Raw Purchases: {pur_df.height:,} rows")

# Check (Trips + Panelists) <-> Purchases Join
tpp_df = tp_df.join(pur_df, on=["panel_year", "trip_code_uc"], how="inner")
print(f"5. Trips + Panelists + Purchases Inner Join: {tpp_df.height:,} rows")

if tpp_df.height == 0:
    print("\n❌ FAILING AT STEP 5: Join with PURCHASES produced 0 rows.")
    print("   Data Type Diagnostics:")
    print("   - Trips trip_code_uc dtype:    ", tp_df["trip_code_uc"].dtype)
    print("   - Purchases trip_code_uc dtype:", pur_df["trip_code_uc"].dtype)
    print("   - Trips panel_year dtype:      ", tp_df["panel_year"].dtype)
    print("   - Purchases panel_year dtype:  ", pur_df["panel_year"].dtype)
    print("\n   Sample Values:")
    print("   - Sample Trips trip_code_uc:    ", tp_df["trip_code_uc"].head(5).to_list())
    print("   - Sample Purchases trip_code_uc:", pur_df["trip_code_uc"].head(5).to_list())
    
    code_overlap = set(tp_df["trip_code_uc"]).intersection(set(pur_df["trip_code_uc"]))
    print(f"   - trip_code_uc overlap (ignoring panel_year): {len(code_overlap)} matching IDs")
    exit(1)

# ---------------------------------------------------------
# STEP 4: Inspect Left Joins (Products, Retailers, Hierarchy)
# ---------------------------------------------------------
print("\nChecking Left Join Targets...")

ret_df = pl.scan_parquet(os.path.join(output_dir, "retailer.parquet")).collect()
print(f"6. Retailer Table Rows: {ret_df.height:,}")

attr_path = os.path.join(output_dir, "product_attr.parquet")
desc_path = os.path.join(output_dir, "product_desc.parquet")

attr_lf = pl.scan_parquet(attr_path)
if "year" in attr_lf.collect_schema().names():
    attr_lf = attr_lf.rename({"year": "panel_year"})

desc_lf = pl.scan_parquet(desc_path)
if "year" in desc_lf.collect_schema().names():
    desc_lf = desc_lf.rename({"year": "panel_year"})

prod_df = attr_lf.join(desc_lf, on=["panel_year", "upc"], how="left", suffix="_prod").collect()
print(f"7. Product Master Table Rows: {prod_df.height:,}")

# Perform full mock master build
master_df = (
    tpp_df.lazy()
    .join(ret_df.lazy(), on="retailer_code", how="left", suffix="_retailer")
    .join(prod_df.lazy(), on=["panel_year", "upc"], how="left", suffix="_product")
    .collect()
)

print(f"\n✅ MASTER TABLE BUILD SUCCESSFUL!")
print(f"   Final Master Dataset Rows: {master_df.height:,}")
print("=" * 60)
