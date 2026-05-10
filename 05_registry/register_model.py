# ─────────────────────────────────────────────────────────────────────────────
# 05_registry/register_model.py
# Run after Training Job — registers model in SageMaker Model Registry
# ─────────────────────────────────────────────────────────────────────────────

import boto3
import sagemaker
from sagemaker import image_uris
from sagemaker.model import Model
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import REGION, SAGEMAKER_ROLE, MODEL_PACKAGE_GROUP

session = sagemaker.Session(boto_session=boto3.Session(region_name=REGION))
sm_client = boto3.client("sagemaker", region_name=REGION)

# ── Read model artifact URI saved by training job ──────────────────────────
artifact_file = os.path.join(
    os.path.dirname(__file__), "..", "04_training", "model_artifact.txt"
)
if not os.path.exists(artifact_file):
    model_artifact = input("Paste model artifact S3 URI (from training output): ").strip()
else:
    with open(artifact_file) as f:
        model_artifact = f.read().strip()

print(f"Registering model: {model_artifact}")

# ── Create Model Package Group (first time only) ───────────────────────────
try:
    sm_client.create_model_package_group(
        ModelPackageGroupName        = MODEL_PACKAGE_GROUP,
        ModelPackageGroupDescription = "XGBoost demand forecast for automobile spare parts",
    )
    print(f"Created model package group: {MODEL_PACKAGE_GROUP}")
except sm_client.exceptions.ClientError as e:
    if "already exists" in str(e) or "AlreadyExists" in str(e):
        print(f"Model package group already exists: {MODEL_PACKAGE_GROUP}")
    else:
        raise

# ── Register the model ─────────────────────────────────────────────────────
container_image = image_uris.retrieve(
    framework = "xgboost",
    region    = REGION,
    version   = "1.5-1",
)

response = sm_client.create_model_package(
    ModelPackageGroupName    = MODEL_PACKAGE_GROUP,
    ModelPackageDescription  = "Spare parts demand forecasting — XGBoost v1 — POC",
    InferenceSpecification   = {
        "Containers": [{
            "Image":           container_image,
            "ModelDataUrl":    model_artifact,
            "Framework":       "XGBOOST",
            "FrameworkVersion":"1.5",
        }],
        "SupportedTransformInstanceTypes":    ["ml.m5.large"],
        "SupportedRealtimeInferenceInstanceTypes": ["ml.m5.large"],
        "SupportedContentTypes":  ["text/csv"],
        "SupportedResponseMIMETypes": ["text/csv"],
    },
    ModelApprovalStatus = "Approved",   # Auto-approve for POC
)

model_package_arn = response["ModelPackageArn"]
print(f"\nModel registered successfully!")
print(f"  ARN: {model_package_arn}")

# Save ARN for batch transform step
with open(os.path.join(os.path.dirname(__file__), "model_package_arn.txt"), "w") as f:
    f.write(model_package_arn)

print("Next step: run 06_batch_transform/run_batch_transform.py")
