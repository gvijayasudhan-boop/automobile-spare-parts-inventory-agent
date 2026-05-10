# ─────────────────────────────────────────────────────────────────────────────
# config.py  –  Shared configuration for all steps
# Update these values before running any step
# ─────────────────────────────────────────────────────────────────────────────

# AWS region where all resources will be created
REGION = "ap-southeast-1"

# Your AWS account ID (12-digit number)
ACCOUNT_ID = "665303623887"  

# IAM role ARN that SageMaker will use (must have SageMaker + S3 + DynamoDB access)
SAGEMAKER_ROLE = "arn:aws:iam::665303623887:role/service-role/AmazonSageMaker-ExecutionRole-20260409T203211"

# S3 bucket for all project data
S3_BUCKET = "auto-parts-inventory"

# S3 folder paths
S3_PATHS = {
    "parts":     f"s3://{S3_BUCKET}/parts/",
    "usage":     f"s3://{S3_BUCKET}/usage/",
    "service":   f"s3://{S3_BUCKET}/service/",
    "supplier":  f"s3://{S3_BUCKET}/supplier/",
    "features":  f"s3://{S3_BUCKET}/features/",
    "forecast":  f"s3://{S3_BUCKET}/forecast/",
    "documents": f"s3://{S3_BUCKET}/documents/",
    "scripts":   f"s3://{S3_BUCKET}/scripts/",
    "model_out": f"s3://{S3_BUCKET}/model-output/",
}

# DynamoDB table name
DYNAMO_TABLE = "AutoPartsInventory"

# Amazon Bedrock model ID (cost-efficient for POC)
BEDROCK_MODEL_ID = "amazon.nova-lite-v1:0"

# S3 Vectors configuration
S3_VECTORS_BUCKET = "auto-parts-vectors"
S3_VECTORS_INDEX  = "parts-policy-index"

# SageMaker experiment name
EXPERIMENT_NAME = "auto-parts-demand-forecast"

# SageMaker model package group
MODEL_PACKAGE_GROUP = "AutoPartsForecasting"

# API Gateway URL (fill in after deploying API Gateway)
API_GATEWAY_URL = "https://YOUR_API_ID.execute-api.ap-southeast-1.amazonaws.com/prod/query"
API_KEY         = "YOUR_API_KEY"  # REPLACE after creating API key in API Gateway
