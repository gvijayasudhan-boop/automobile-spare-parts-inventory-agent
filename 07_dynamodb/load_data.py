# ─────────────────────────────────────────────────────────────────────────────
# 07_dynamodb/load_data.py
# Loads current_inventory.csv into DynamoDB using batch_writer
# Run after create_table.py
# ─────────────────────────────────────────────────────────────────────────────

import boto3
import pandas as pd
from decimal import Decimal
import sys, os, math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import DYNAMO_TABLE, S3_BUCKET, REGION

s3     = boto3.client("s3", region_name=REGION)
dynamo = boto3.resource("dynamodb", region_name=REGION)
table  = dynamo.Table(DYNAMO_TABLE)

def safe_decimal(val):
    """Convert float to Decimal safely for DynamoDB."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return Decimal("0")
    return Decimal(str(round(float(val), 2)))

def load_from_s3():
    """Download inventory CSV from S3 and load into DynamoDB."""
    import io
    print(f"Downloading current_inventory.csv from S3...")
    obj = s3.get_object(Bucket=S3_BUCKET, Key="parts/current_inventory.csv")
    df  = pd.read_csv(io.BytesIO(obj["Body"].read()))
    return df

def load_from_local():
    """Fallback: load from local data folder."""
    local_path = os.path.join(
        os.path.dirname(__file__), "..", "01_data_prep", "data", "current_inventory.csv"
    )
    print(f"Loading from local: {local_path}")
    return pd.read_csv(local_path)

def load_to_dynamo(df):
    print(f"Loading {len(df)} items into DynamoDB table '{DYNAMO_TABLE}'...")
    loaded = 0
    skipped = 0

    with table.batch_writer() as batch:
        for _, row in df.iterrows():
            try:
                item = {
                    # ── Composite Key ──────────────────────────────────────
                    "part_id":        str(row["part_id"]),
                    "sk":             str(row["sk"]),           # vehicle_model#variant

                    # ── Attributes ─────────────────────────────────────────
                    "vehicle_model":  str(row["vehicle_model"]),
                    "variant":        str(row["variant"]),
                    "part_name":      str(row["part_name"]),
                    "category":       str(row["category"]),
                    "current_stock":  int(row["current_stock"]),
                    "reorder_level":  int(row["reorder_level"]),
                    "supplier_id":    str(row["supplier_id"]),
                    "lead_time_days": int(row["lead_time_days"]),
                    "moq":            int(row["moq"]),
                    "unit_cost":      safe_decimal(row["unit_cost"]),
                    "single_source":  bool(row.get("single_source", False)),
                    "shortage_flag":  int(row["current_stock"]) < int(row["reorder_level"]),
                }
                batch.put_item(Item=item)
                loaded += 1
            except Exception as e:
                print(f"  Skipped row ({row.get('part_id','?')}): {e}")
                skipped += 1

    print(f"Load complete — Loaded: {loaded}, Skipped: {skipped}")

def verify_load():
    """Quick verification — query one item."""
    print("\nVerifying load...")
    response = table.query(
        KeyConditionExpression=boto3.dynamodb.conditions.Key("part_id").eq("PART-BRK-PAD-F"),
        Limit=3
    )
    items = response.get("Items", [])
    print(f"  Sample query (PART-BRK-PAD-F): {len(items)} items found")
    for item in items[:2]:
        print(f"    {item['part_id']} | {item['sk']} | stock={item['current_stock']} | reorder={item['reorder_level']}")

if __name__ == "__main__":
    # Try S3 first, fall back to local
    try:
        df = load_from_s3()
    except Exception as e:
        print(f"S3 load failed ({e}), trying local...")
        df = load_from_local()

    print(f"  Rows to load: {len(df)}")
    load_to_dynamo(df)
    verify_load()
    print("\nNext step: deploy 08_lambda/lambda_function.py to AWS Lambda")
