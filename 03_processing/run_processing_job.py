# ─────────────────────────────────────────────────────────────────────────────
# 03_processing/run_processing_job.py
# Run from SageMaker Studio Notebook (Python 3 kernel)
# pip install sagemaker boto3
# ─────────────────────────────────────────────────────────────────────────────

import boto3
import sagemaker
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import S3_BUCKET, REGION, SAGEMAKER_ROLE

from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput

session = sagemaker.Session(boto_session=boto3.Session(region_name=REGION))

print("Starting SageMaker Processing Job...")
print(f"  Bucket : s3://{S3_BUCKET}")
print(f"  Region : {REGION}")

processor = SKLearnProcessor(
    framework_version = "0.23-1",
    role              = SAGEMAKER_ROLE,
    instance_type     = "ml.t3.medium",   # cost-efficient for POC
    instance_count    = 1,
    sagemaker_session = session,
    base_job_name     = "auto-parts-processing",
)

processor.run(
    code = f"s3://{S3_BUCKET}/scripts/preprocessing.py",

    inputs = [
        ProcessingInput(
            source      = f"s3://{S3_BUCKET}/usage/",
            destination = "/opt/ml/processing/input/usage/",
            input_name  = "usage",
        ),
        ProcessingInput(
            source      = f"s3://{S3_BUCKET}/parts/",
            destination = "/opt/ml/processing/input/parts/",
            input_name  = "parts",
        ),
    ],

    outputs = [
        ProcessingOutput(
            source      = "/opt/ml/processing/output/train/",
            destination = f"s3://{S3_BUCKET}/features/train/",
            output_name = "train",
        ),
        ProcessingOutput(
            source      = "/opt/ml/processing/output/val/",
            destination = f"s3://{S3_BUCKET}/features/val/",
            output_name = "val",
        ),
        ProcessingOutput(
            source      = "/opt/ml/processing/output/inference/",
            destination = f"s3://{S3_BUCKET}/features/inference/",
            output_name = "inference",
        ),
    ],

    wait = True,
    logs = True,
)

print("\nProcessing Job complete!")
print(f"Features stored at: s3://{S3_BUCKET}/features/")
print("Next step: run 04_training/run_training_job.py")
