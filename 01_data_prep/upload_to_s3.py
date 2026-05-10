# ─────────────────────────────────────────────────────────────────────────────
# 01_data_prep/upload_to_s3.py
# Run after generate_data.py  –  uploads all CSVs to S3
# ─────────────────────────────────────────────────────────────────────────────

import boto3
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import S3_BUCKET, REGION

s3 = boto3.client("s3", region_name=REGION)

def create_bucket_if_not_exists():
    try:
        s3.head_bucket(Bucket=S3_BUCKET)
        print(f"Bucket s3://{S3_BUCKET} already exists.")
    except Exception:
        if REGION == "us-east-1":
            s3.create_bucket(Bucket=S3_BUCKET)
        else:
            s3.create_bucket(
                Bucket=S3_BUCKET,
                CreateBucketConfiguration={"LocationConstraint": REGION}
            )
        # Block all public access
        s3.put_public_access_block(
            Bucket=S3_BUCKET,
            PublicAccessBlockConfiguration={
                "BlockPublicAcls": True, "IgnorePublicAcls": True,
                "BlockPublicPolicy": True, "RestrictPublicBuckets": True
            }
        )
        print(f"Created bucket s3://{S3_BUCKET}")

def upload_file(local_path, s3_key):
    s3.upload_file(local_path, S3_BUCKET, s3_key)
    print(f"  Uploaded: {local_path} → s3://{S3_BUCKET}/{s3_key}")

if __name__ == "__main__":
    create_bucket_if_not_exists()

    data_dir = os.path.join(os.path.dirname(__file__), "data")

    uploads = [
        (f"{data_dir}/parts_catalog.csv",   "parts/parts_catalog.csv"),
        (f"{data_dir}/usage_history.csv",   "usage/usage_history.csv"),
        (f"{data_dir}/service_schedule.csv","service/service_schedule.csv"),
        (f"{data_dir}/supplier_master.csv", "supplier/supplier_master.csv"),
        (f"{data_dir}/current_inventory.csv","parts/current_inventory.csv"),
    ]

    # Also upload the preprocessing script (needed by Processing Job)
    script_path = os.path.join(os.path.dirname(__file__), "..", "03_processing", "preprocessing.py")
    uploads.append((script_path, "scripts/preprocessing.py"))

    print("\nUploading files to S3...")
    for local, key in uploads:
        if os.path.exists(local):
            upload_file(local, key)
        else:
            print(f"  WARNING: {local} not found — run generate_data.py first")

    print("\nAll uploads complete.")
    print(f"Data lake: s3://{S3_BUCKET}/")
