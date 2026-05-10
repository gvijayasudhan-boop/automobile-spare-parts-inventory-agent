# ML + GenAI Automobile Spare Parts Inventory Agent
## Complete Setup Guide — Run Order

---

## Prerequisites

### Install Python packages
```bash
pip install pandas numpy boto3 sagemaker streamlit requests
```

### Configure AWS credentials
```bash
aws configure
# Enter: Access Key ID, Secret Access Key, Region (ap-southeast-1), Output format (json)
```

### Update config.py
Open `config.py` and update:
- `ACCOUNT_ID`      — your 12-digit AWS account ID
- `SAGEMAKER_ROLE`  — ARN of your SageMaker execution IAM role
- (After deploying API Gateway) `API_GATEWAY_URL` and `API_KEY`

---

## Step-by-Step Run Order

### STEP 1 — Generate & Upload Data
```bash
cd 01_data_prep/
python generate_data.py        # Creates ./data/ folder with all CSV files
python upload_to_s3.py         # Uploads CSVs + preprocessing script to S3
```

### STEP 2 — Upload is handled inside Step 1 above (no separate step needed)

### STEP 3 — Feature Engineering (SageMaker Processing Job)
```bash
cd ../03_processing/
python run_processing_job.py   # Launches managed SageMaker job (~10-15 min)
```
- Input:  s3://auto-parts-inventory/usage/ and /parts/
- Output: s3://auto-parts-inventory/features/train/, /val/, /inference/

### STEP 4 — Model Training (SageMaker Training Job)
```bash
cd ../04_training/
python run_training_job.py     # Trains XGBoost model (~10-20 min)
```
- Output: s3://auto-parts-inventory/model-output/
- Saves:  model_artifact.txt (URI of trained model)

### STEP 5 — Model Registry
```bash
cd ../05_registry/
python register_model.py       # Registers model in SageMaker Model Registry
```
- Reads: model_artifact.txt from Step 4
- Saves: model_package_arn.txt

### STEP 6 — Batch Transform (Inference)
```bash
cd ../06_batch_transform/
python run_batch_transform.py  # Runs batch predictions (~5-10 min)
```
- Input:  s3://auto-parts-inventory/features/inference/
- Output: s3://auto-parts-inventory/forecast/forecast_output.csv

### STEP 7 — DynamoDB Setup + Load
```bash
cd ../07_dynamodb/
python create_table.py         # Creates table + 3 GSIs (runs in seconds)
python load_data.py            # Loads parts inventory from S3 into DynamoDB
```

### STEP 8 — Lambda Deployment
1. Go to AWS Console → Lambda → Create Function
2. Runtime: Python 3.12
3. Paste contents of `08_lambda/lambda_function.py`
4. Set environment variables (see top of lambda_function.py)
5. Set timeout: 30 seconds, Memory: 512 MB
6. Add IAM permissions: DynamoDB, S3, Bedrock

### STEP 9 — Bedrock is called from Lambda (no separate setup needed)
- Just ensure your Lambda IAM role has: `bedrock:InvokeModel` permission
- Model IDs: amazon.nova-lite-v1:0 and amazon.titan-embed-text-v1

### STEP 10 — RAG Embedding (run once)
```bash
cd ../10_rag/
python embed_documents.py      # Creates S3 Vectors index + embeds policy docs
```

### STEP 11 — API Gateway (AWS Console)
1. Go to API Gateway → Create REST API
2. Create resource: /query  →  Method: POST
3. Integration: Lambda Function → select AutoPartsInventoryAgent
4. Enable API key required
5. Deploy to stage: prod
6. Copy the Invoke URL → update API_GATEWAY_URL in config.py

### STEP 11 — Run Streamlit UI
```bash
cd ../11_ui/
streamlit run app.py
```
Opens at: http://localhost:8501

---

## Lambda IAM Role Permissions Required

Attach these policies to your Lambda execution role:
- AmazonDynamoDBFullAccess (or specific table access)
- AmazonS3ReadOnlyAccess
- AmazonBedrockFullAccess
- AmazonS3VectorsFullAccess (if using RAG)

---

## Cost Reminder (Sandbox SOP)

| Service         | Action after demo                    |
|----------------|--------------------------------------|
| SageMaker       | Delete models, endpoints, jobs       |
| DynamoDB        | Delete table (or leave — on-demand)  |
| S3              | Delete /features/ and /forecast/     |
| Lambda          | Leave (no idle cost)                 |
| API Gateway     | Delete stage after demo              |
| S3 Vectors      | Delete index and bucket              |

MANDATORY TAGS on every resource: Owner · ISU · Project · Purpose · Group Name
