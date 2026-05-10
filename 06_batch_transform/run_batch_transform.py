# ─────────────────────────────────────────────────────────────────────────────
# 06_batch_transform/run_batch_transform.py
# Runs batch predictions — output written to S3/forecast/
# No real-time endpoint needed (no idle charges!)
# ─────────────────────────────────────────────────────────────────────────────

import boto3
import sagemaker
from sagemaker import image_uris
from sagemaker.transformer import Transformer
from sagemaker.model import Model
import sys, os, time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import S3_BUCKET, REGION, SAGEMAKER_ROLE

session   = sagemaker.Session(boto_session=boto3.Session(region_name=REGION))
sm_client = boto3.client("sagemaker", region_name=REGION)

# ── Read model artifact URI ────────────────────────────────────────────────
artifact_file = os.path.join(
    os.path.dirname(__file__), "..", "04_training", "model_artifact.txt"
)
with open(artifact_file) as f:
    model_artifact = f.read().strip()

container_image = image_uris.retrieve(
    framework = "xgboost",
    region    = REGION,
    version   = "1.5-1",
)

# ── Create a SageMaker Model object ───────────────────────────────────────
model_name = f"auto-parts-model-{int(time.time())}"

sm_client.create_model(
    ModelName        = model_name,
    ExecutionRoleArn = SAGEMAKER_ROLE,
    PrimaryContainer = {
        "Image":        container_image,
        "ModelDataUrl": model_artifact,
    }
)
print(f"Created SageMaker model: {model_name}")

# ── Create Transformer (Batch Transform) ──────────────────────────────────
transformer = Transformer(
    model_name        = model_name,
    instance_count    = 1,
    instance_type     = "ml.m5.large",
    output_path       = f"s3://{S3_BUCKET}/forecast/raw/",
    sagemaker_session = session,
    base_transform_job_name = "auto-parts-forecast",
    strategy          = "MultiRecord",
    assemble_with     = "Line",
)

print("Starting Batch Transform Job...")
print(f"  Input : s3://{S3_BUCKET}/features/inference/inference.csv")
print(f"  Output: s3://{S3_BUCKET}/forecast/raw/")

transformer.transform(
    data         = f"s3://{S3_BUCKET}/features/inference/inference.csv",
    content_type = "text/csv",
    split_type   = "Line",
    wait         = True,
    logs         = True,
)

print("\nBatch Transform complete!")
print(f"Raw predictions: s3://{S3_BUCKET}/forecast/raw/")

# ── Merge predictions with metadata ───────────────────────────────────────
print("\nMerging predictions with part metadata...")
import boto3
import pandas as pd
import io

s3 = boto3.client("s3", region_name=REGION)

# Download inference metadata
meta_obj = s3.get_object(Bucket=S3_BUCKET, Key="features/inference/inference_meta.csv")
df_meta  = pd.read_csv(io.BytesIO(meta_obj["Body"].read()))

# Download raw predictions
pred_key = "forecast/raw/inference.csv.out"
try:
    pred_obj = s3.get_object(Bucket=S3_BUCKET, Key=pred_key)
    preds    = [float(line.strip()) for line in pred_obj["Body"].read().decode().strip().split("\n") if line.strip()]
    df_meta["predicted_demand"] = preds
    df_meta["predicted_demand"] = df_meta["predicted_demand"].clip(lower=0).round(1)

    # Save merged forecast
    buf = io.StringIO()
    df_meta.to_csv(buf, index=False)
    s3.put_object(
        Bucket = S3_BUCKET,
        Key    = "forecast/forecast_output.csv",
        Body   = buf.getvalue().encode(),
    )
    print(f"Forecast saved: s3://{S3_BUCKET}/forecast/forecast_output.csv")
    print(f"Sample predictions:\n{df_meta.head()}")
except Exception as e:
    print(f"Metadata merge skipped (run manually if needed): {e}")

print("\nNext step: run 07_dynamodb/create_table.py")
