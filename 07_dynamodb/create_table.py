# ─────────────────────────────────────────────────────────────────────────────
# 07_dynamodb/create_table.py
# Creates DynamoDB table with composite key + 3 GSIs
# Run once before loading data
# ─────────────────────────────────────────────────────────────────────────────

import boto3
import sys, os, time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import DYNAMO_TABLE, REGION

dynamo = boto3.client("dynamodb", region_name=REGION)

def create_table():
    try:
        dynamo.describe_table(TableName=DYNAMO_TABLE)
        print(f"Table '{DYNAMO_TABLE}' already exists — skipping creation.")
        return
    except dynamo.exceptions.ResourceNotFoundException:
        pass

    print(f"Creating DynamoDB table: {DYNAMO_TABLE}")
    dynamo.create_table(
        TableName            = DYNAMO_TABLE,
        BillingMode          = "PAY_PER_REQUEST",  # on-demand — no idle cost

        # ── Composite Key ────────────────────────────────────────────────────
        KeySchema = [
            { "AttributeName": "part_id", "KeyType": "HASH"  },  # Partition Key
            { "AttributeName": "sk",      "KeyType": "RANGE" },  # Sort Key: vehicle_model#variant
        ],
        AttributeDefinitions = [
            { "AttributeName": "part_id",       "AttributeType": "S" },
            { "AttributeName": "sk",            "AttributeType": "S" },
            { "AttributeName": "vehicle_model", "AttributeType": "S" },
            { "AttributeName": "category",      "AttributeType": "S" },
            { "AttributeName": "supplier_id",   "AttributeType": "S" },
            { "AttributeName": "current_stock", "AttributeType": "N" },
            { "AttributeName": "lead_time_days","AttributeType": "N" },
        ],

        # ── Global Secondary Indexes ─────────────────────────────────────────
        GlobalSecondaryIndexes = [
            {
                # GSI-1: Query all parts for a given vehicle model
                "IndexName": "VehicleModelIndex",
                "KeySchema": [
                    { "AttributeName": "vehicle_model", "KeyType": "HASH"  },
                    { "AttributeName": "part_id",       "KeyType": "RANGE" },
                ],
                "Projection": { "ProjectionType": "ALL" },
            },
            {
                # GSI-2: Query all parts in a category sorted by stock (low stock first)
                "IndexName": "CategoryStockIndex",
                "KeySchema": [
                    { "AttributeName": "category",      "KeyType": "HASH"  },
                    { "AttributeName": "current_stock", "KeyType": "RANGE" },
                ],
                "Projection": { "ProjectionType": "ALL" },
            },
            {
                # GSI-3: Query all parts by supplier sorted by lead time
                "IndexName": "SupplierIndex",
                "KeySchema": [
                    { "AttributeName": "supplier_id",    "KeyType": "HASH"  },
                    { "AttributeName": "lead_time_days", "KeyType": "RANGE" },
                ],
                "Projection": { "ProjectionType": "ALL" },
            },
        ],
    )

    # Wait for table to be active
    print("Waiting for table to become ACTIVE...")
    waiter = dynamo.get_waiter("table_exists")
    waiter.wait(TableName=DYNAMO_TABLE)
    print(f"Table '{DYNAMO_TABLE}' is ACTIVE.")
    print(f"  PK  : part_id (String)")
    print(f"  SK  : sk = vehicle_model#variant (String)")
    print(f"  GSI1: VehicleModelIndex  — all parts for a vehicle")
    print(f"  GSI2: CategoryStockIndex — consumables with low stock")
    print(f"  GSI3: SupplierIndex      — parts by supplier + lead time")

if __name__ == "__main__":
    create_table()
    print("\nNext step: run 07_dynamodb/load_data.py")
