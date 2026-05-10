# ─────────────────────────────────────────────────────────────────────────────
# 01_data_prep/generate_data.py
# Run this locally or in SageMaker Studio Notebook (Python 3 kernel)
# pip install pandas numpy faker
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import random

random.seed(42)
np.random.seed(42)

OUTPUT_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Reference data ─────────────────────────────────────────────────────────
VEHICLE_MODELS = [
    ("MARUTI_SWIFT",   "2021"), ("MARUTI_SWIFT",   "2023"),
    ("HONDA_CITY",     "2022"), ("HONDA_CITY",     "2020"),
    ("HYUNDAI_I20",    "2023"), ("HYUNDAI_I20",    "2021"),
    ("TOYOTA_INNOVA",  "2020"), ("TOYOTA_INNOVA",  "2022"),
    ("TATA_NEXON",     "2023"), ("KIA_SELTOS",     "2022"),
]

PARTS = [
    # (part_id, part_name, category, compatible_models, base_demand, seasonal_factor)
    ("PART-OIL-5W30",  "Engine Oil 5W-30 1L",    "Consumable",  "ALL",         40, "none"),
    ("PART-OIL-10W40", "Engine Oil 10W-40 1L",   "Consumable",  "ALL",         20, "none"),
    ("PART-AIR-FLT",   "Air Filter",             "Consumable",  "ALL",         15, "summer"),
    ("PART-OIL-FLT",   "Oil Filter",             "Consumable",  "ALL",         35, "none"),
    ("PART-BRK-PAD-F", "Brake Pad Front",        "Wear Item",   "ALL",         12, "monsoon"),
    ("PART-BRK-PAD-R", "Brake Pad Rear",         "Wear Item",   "ALL",         8,  "monsoon"),
    ("PART-TYRE-185",  "Tyre 185/65 R15",        "Wear Item",   "SMALL",       5,  "monsoon"),
    ("PART-TYRE-195",  "Tyre 195/65 R15",        "Wear Item",   "MEDIUM",      4,  "monsoon"),
    ("PART-TYRE-215",  "Tyre 215/60 R17",        "Wear Item",   "LARGE",       3,  "monsoon"),
    ("PART-BATT-35AH", "Battery 35Ah",           "Consumable",  "SMALL",       6,  "summer"),
    ("PART-BATT-45AH", "Battery 45Ah",           "Consumable",  "MEDIUM",      5,  "summer"),
    ("PART-BATT-60AH", "Battery 60Ah",           "Consumable",  "LARGE",       3,  "summer"),
    ("PART-SPRK-NGK",  "Spark Plug NGK",         "Consumable",  "PETROL",      20, "none"),
    ("PART-CLNT-1L",   "Coolant 1L",             "Consumable",  "ALL",         10, "summer"),
    ("PART-WSHLD-FLD", "Windshield Fluid 1L",    "Consumable",  "ALL",         8,  "monsoon"),
    ("PART-BRK-FLUID", "Brake Fluid DOT4",       "Consumable",  "ALL",         6,  "none"),
    ("PART-ACBELT",    "AC Belt",                "Wear Item",   "ALL",         4,  "summer"),
    ("PART-ALTBELT",   "Alternator Belt",        "Wear Item",   "ALL",         3,  "none"),
    ("PART-TIMCHAIN",  "Timing Chain",           "OEM Part",    "ALL",         2,  "none"),
    ("PART-CLUTCH",    "Clutch Plate",           "Wear Item",   "MANUAL",      3,  "none"),
    ("PART-FUELFLTR",  "Fuel Filter",            "Consumable",  "ALL",         5,  "none"),
    ("PART-SUSPARM",   "Suspension Arm",         "OEM Part",    "ALL",         2,  "monsoon"),
    ("PART-SHOCKABS",  "Shock Absorber",         "Wear Item",   "ALL",         3,  "monsoon"),
    ("PART-WIPERBLD",  "Wiper Blade",            "Consumable",  "ALL",         8,  "monsoon"),
    ("PART-HEADLAMP",  "Headlamp Bulb H4",       "Consumable",  "ALL",         6,  "none"),
]

SUPPLIERS = [
    ("SUP-001", "Castrol India",      3,  12),
    ("SUP-002", "Honda OEM Supplies", 10, 25),
    ("SUP-003", "Maruti Suzuki OEM",  7,  20),
    ("SUP-004", "Hyundai OEM",        8,  22),
    ("SUP-005", "MRF Tyres",          14, 30),
    ("SUP-006", "Amaron Batteries",   3,  10),
    ("SUP-007", "NGK Spark Plugs",    6,  15),
    ("SUP-008", "Bosch India",        5,  18),
    ("SUP-009", "Toyota OEM",         10, 25),
    ("SUP-010", "Universal Auto",     4,  8),
]

VEHICLE_SIZE = {
    "MARUTI_SWIFT":  "SMALL",  "HONDA_CITY":    "MEDIUM",
    "HYUNDAI_I20":   "SMALL",  "TOYOTA_INNOVA": "LARGE",
    "TATA_NEXON":    "MEDIUM", "KIA_SELTOS":    "MEDIUM",
}

VEHICLE_TYPE = {
    "MARUTI_SWIFT":  "PETROL",  "HONDA_CITY":    "PETROL",
    "HYUNDAI_I20":   "PETROL",  "TOYOTA_INNOVA": "DIESEL",
    "TATA_NEXON":    "PETROL",  "KIA_SELTOS":    "PETROL",
}

VEHICLE_TRANS = {
    "MARUTI_SWIFT":  "MANUAL",  "HONDA_CITY":    "AUTOMATIC",
    "HYUNDAI_I20":   "MANUAL",  "TOYOTA_INNOVA": "MANUAL",
    "TATA_NEXON":    "AUTOMATIC","KIA_SELTOS":   "AUTOMATIC",
}

def is_compatible(part_compat, vehicle_model):
    size  = VEHICLE_SIZE.get(vehicle_model, "MEDIUM")
    vtype = VEHICLE_TYPE.get(vehicle_model, "PETROL")
    vtrans= VEHICLE_TRANS.get(vehicle_model, "MANUAL")
    if part_compat == "ALL":    return True
    if part_compat == size:     return True
    if part_compat == vtype:    return True
    if part_compat == vtrans:   return True
    return False

def seasonal_multiplier(month, factor):
    if factor == "none":    return 1.0
    if factor == "monsoon": return 1.6 if month in [6,7,8,9] else 1.0
    if factor == "summer":  return 1.5 if month in [3,4,5] else 1.0
    return 1.0

# ── 1. Parts Catalog ──────────────────────────────────────────────────────────
print("Generating parts catalog...")
catalog_rows = []
for pid, pname, cat, compat, base_demand, sfactor in PARTS:
    for vmodel, vvariant in VEHICLE_MODELS:
        if not is_compatible(compat, vmodel):
            continue
        sup = random.choice(SUPPLIERS)
        reorder = max(5, int(base_demand * 0.4))
        stock   = random.randint(reorder, reorder * 6)
        catalog_rows.append({
            "part_id":        pid,
            "part_name":      pname,
            "vehicle_model":  vmodel,
            "variant":        vvariant,
            "category":       cat,
            "current_stock":  stock,
            "reorder_level":  reorder,
            "supplier_id":    sup[0],
            "supplier_name":  sup[1],
            "lead_time_days": sup[2],
            "moq":            sup[3],
            "unit_cost":      round(random.uniform(50, 5000), 2),
            "single_source":  random.choice([True, False, False, False]),
            "sk":             f"{vmodel}#{vvariant}",
        })

df_catalog = pd.DataFrame(catalog_rows)
df_catalog.to_csv(f"{OUTPUT_DIR}/parts_catalog.csv", index=False)
print(f"  parts_catalog.csv: {len(df_catalog)} rows")

# ── 2. Usage History (18 months) ─────────────────────────────────────────────
print("Generating usage history...")
start_date = datetime(2023, 1, 1)
end_date   = datetime(2024, 6, 30)
usage_rows = []

for pid, pname, cat, compat, base_demand, sfactor in PARTS:
    for vmodel, vvariant in VEHICLE_MODELS:
        if not is_compatible(compat, vmodel):
            continue
        current = start_date
        while current <= end_date:
            month = current.month
            sm    = seasonal_multiplier(month, sfactor)
            weekly_demand = max(1, int(np.random.poisson(base_demand / 4 * sm)))
            for day_offset in range(0, 7):
                service_date = current + timedelta(days=day_offset)
                if service_date > end_date:
                    break
                if random.random() < 0.3:
                    qty = max(1, int(np.random.poisson(weekly_demand / 3)))
                    usage_rows.append({
                        "part_id":       pid,
                        "vehicle_model": vmodel,
                        "variant":       vvariant,
                        "service_date":  service_date.strftime("%Y-%m-%d"),
                        "qty_used":      qty,
                        "service_type":  random.choice(["Periodic","Periodic","Repair","Recall","Campaign"]),
                        "week_number":   service_date.isocalendar()[1],
                        "month":         service_date.month,
                        "year":          service_date.year,
                    })
            current += timedelta(weeks=1)

df_usage = pd.DataFrame(usage_rows)
df_usage.to_csv(f"{OUTPUT_DIR}/usage_history.csv", index=False)
print(f"  usage_history.csv: {len(df_usage)} rows")

# ── 3. Service Schedule (next 30 days) ───────────────────────────────────────
print("Generating service schedule...")
sched_rows = []
today = datetime.now()
for i in range(200):
    vm, vv = random.choice(VEHICLE_MODELS)
    appt_date = today + timedelta(days=random.randint(1, 30))
    sched_rows.append({
        "vehicle_id":      f"VEH-{random.randint(1000,9999)}",
        "vehicle_model":   vm,
        "variant":         vv,
        "appointment_date":appt_date.strftime("%Y-%m-%d"),
        "service_type":    random.choice(["Periodic","Repair","Inspection"]),
        "km_reading":      random.randint(10000, 120000),
    })

df_sched = pd.DataFrame(sched_rows)
df_sched.to_csv(f"{OUTPUT_DIR}/service_schedule.csv", index=False)
print(f"  service_schedule.csv: {len(df_sched)} rows")

# ── 4. Supplier Master ────────────────────────────────────────────────────────
print("Generating supplier master...")
sup_rows = []
for sid, sname, lead, moq in SUPPLIERS:
    sup_rows.append({
        "supplier_id":   sid,
        "supplier_name": sname,
        "lead_time_days":lead,
        "moq":           moq,
        "contact_email": f"orders@{sname.lower().replace(' ','-')}.com",
        "payment_terms": random.choice(["Net30","Net45","Advance"]),
        "active":        True,
    })

df_sup = pd.DataFrame(sup_rows)
df_sup.to_csv(f"{OUTPUT_DIR}/supplier_master.csv", index=False)
print(f"  supplier_master.csv: {len(df_sup)} rows")

# ── 5. Current Inventory Snapshot (for DynamoDB load) ────────────────────────
# Already in parts_catalog — create a clean snapshot version
df_snapshot = df_catalog[[
    "part_id","sk","vehicle_model","variant","current_stock",
    "reorder_level","supplier_id","lead_time_days","category","unit_cost",
    "moq","single_source","part_name"
]].copy()
df_snapshot.to_csv(f"{OUTPUT_DIR}/current_inventory.csv", index=False)
print(f"  current_inventory.csv: {len(df_snapshot)} rows")

print("\nAll data files generated in ./data/")
print("Next: run upload_to_s3.py to push files to S3.")
