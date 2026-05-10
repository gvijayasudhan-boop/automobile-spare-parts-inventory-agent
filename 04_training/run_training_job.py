# ─────────────────────────────────────────────────────────────────────────────
# 04_training/run_training_job.py
# Run from SageMaker Studio Notebook after Processing Job completes
# ─────────────────────────────────────────────────────────────────────────────

import boto3
import sagemaker
from sagemaker import image_uris
from sagemaker.inputs import TrainingInput
from sagemaker.experiments import Run
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import S3_BUCKET, REGION, SAGEMAKER_ROLE, EXPERIMENT_NAME

session = sagemaker.Session(boto_session=boto3.Session(region_name=REGION))

# ── Get XGBoost built-in container ────────────────────────────────────────
container = image_uris.retrieve(
    framework  = "xgboost",
    region     = REGION,
    version    = "1.5-1",
)

print("Starting SageMaker Training Job...")
print(f"  Container : {container}")
print(f"  Bucket    : s3://{S3_BUCKET}")

# ── Estimator ──────────────────────────────────────────────────────────────
estimator = sagemaker.estimator.Estimator(
    image_uri         = container,
    role              = SAGEMAKER_ROLE,
    instance_count    = 1,
    instance_type     = "ml.m5.large",
    output_path       = f"s3://{S3_BUCKET}/model-output/",
    sagemaker_session = session,
    base_job_name     = "auto-parts-xgboost",
)

# ── Hyperparameters ────────────────────────────────────────────────────────
estimator.set_hyperparameters(
    objective         = "reg:squarederror",
    num_round         = 150,
    max_depth         = 6,
    eta               = 0.1,
    subsample         = 0.8,
    colsample_bytree  = 0.8,
    min_child_weight  = 1,
    eval_metric       = "rmse",
    early_stopping_rounds = 20,
)

# ── Data channels ──────────────────────────────────────────────────────────
train_input = TrainingInput(
    s3_data      = f"s3://{S3_BUCKET}/features/train/",
    content_type = "text/csv",
)
val_input = TrainingInput(
    s3_data      = f"s3://{S3_BUCKET}/features/val/",
    content_type = "text/csv",
)

# ── Train (with Experiments tracking) ─────────────────────────────────────
try:
    # SageMaker Experiments (if available in your SDK version)
    with Run(
        experiment_name = EXPERIMENT_NAME,
        run_name        = "xgboost-run-01",
        sagemaker_session = session,
    ) as run:
        estimator.fit(
            inputs = {"train": train_input, "validation": val_input},
            wait   = True,
            logs   = True,
        )
        run.log_metric("best_train_rmse", estimator.training_job_analytics().dataframe()["value"].min())
except Exception:
    # Fallback without Experiments
    estimator.fit(
        inputs = {"train": train_input, "validation": val_input},
        wait   = True,
        logs   = True,
    )

print("\nTraining Job complete!")
print(f"Model artifact: {estimator.model_data}")
print("Next step: run 05_registry/register_model.py")

# Save model URI for next steps
model_artifact = estimator.model_data
with open(os.path.join(os.path.dirname(__file__), "model_artifact.txt"), "w") as f:
    f.write(model_artifact)
print(f"Model URI saved to model_artifact.txt")
