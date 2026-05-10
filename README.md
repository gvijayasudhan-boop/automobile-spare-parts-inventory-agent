# 🚗 ML + GenAI Powered Automobile Spare Parts Inventory Agent

[![AWS](https://img.shields.io/badge/AWS-232F3E?style=for-the-badge&logo=amazon-aws&logoColor=white)](https://aws.amazon.com)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Amazon SageMaker](https://img.shields.io/badge/Amazon%20SageMaker-FF9900?style=for-the-badge&logo=amazon-aws&logoColor=white)](https://aws.amazon.com/sagemaker)
[![Amazon Bedrock](https://img.shields.io/badge/Amazon%20Bedrock-FF9900?style=for-the-badge&logo=amazon-aws&logoColor=white)](https://aws.amazon.com/bedrock)
[![DynamoDB](https://img.shields.io/badge/Amazon%20DynamoDB-4053D6?style=for-the-badge&logo=amazon-dynamodb&logoColor=white)](https://aws.amazon.com/dynamodb)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)

---

## 📋 Overview

An intelligent **ML + GenAI powered chatbot** that helps automobile service centers manage spare parts inventory, predict demand, and generate restock recommendations — all through natural language conversation.

> **"Which parts are critically low for Honda City this week?"**
> The agent answers instantly — with ML-driven restock quantities, supplier details, and clear explanations.

---

## 🎯 Problem Statement

Automobile service centers face:
- ❌ Critical parts running out unexpectedly — halting scheduled services
- ❌ Static reorder rules ignoring seasonal demand spikes
- ❌ No conversational way to query inventory in natural language
- ❌ Manual procurement decisions without ML demand forecasts
- ❌ OEM fitment guides and supplier SOPs scattered across systems

---

## ✅ Solution

A complete **11-step ML + GenAI pipeline** on AWS that:
- 🔮 **Predicts** parts demand 30-90 days ahead using XGBoost
- 📊 **Recommends** exact restock quantities with explainable formulas
- 💬 **Answers** natural language queries conversationally
- 📄 **Grounds** policy answers in official OEM documents via RAG
- 🛡️ **Handles** all queries gracefully — never shows errors to users

---

## 🏗️ Architecture — 11 Steps

```
Step 1:  Microsoft Copilot + SageMaker Studio  →  Data Generation & Cleaning
Step 2:  Amazon S3                             →  Central Data Lake
Step 3:  SageMaker Processing Job             →  Feature Engineering
Step 4:  SageMaker Training Job               →  XGBoost Demand Forecasting
Step 5:  SageMaker Model Registry             →  Model Versioning & Governance
Step 6:  SageMaker Batch Transform            →  Offline Batch Predictions
Step 7:  Amazon DynamoDB                      →  Real-Time Parts Inventory
Step 8:  AWS Lambda                           →  Restock Logic + Out-of-Scope Handling
Step 9:  Amazon Bedrock (Nova Lite APAC)      →  Conversational AI Explanation
Step 10: Amazon S3 Vectors (RAG)              →  OEM & Policy Document Retrieval
Step 11: API Gateway + Streamlit UI           →  Chat Interface
```

---

## 🛠️ Technology Stack

| Layer | Service | Purpose |
|-------|---------|---------|
| Data Preparation | Microsoft Copilot + SageMaker Studio | Generate & clean synthetic data |
| Data Storage | Amazon S3 | Central data lake |
| Parts Inventory | Amazon DynamoDB | Real-time inventory with composite key |
| Feature Engineering | SageMaker Processing Job | Lag features, seasonality, rolling averages |
| ML Training | SageMaker Training Job (XGBoost) | Demand forecasting model |
| Model Governance | SageMaker Model Registry | Versioning & approval workflow |
| ML Inference | SageMaker Batch Transform | Offline batch predictions |
| Business Logic | AWS Lambda | Restock formula + query routing |
| GenAI | Amazon Bedrock (Nova Lite APAC) | Conversational AI responses |
| RAG | Amazon S3 Vectors | OEM fitment guides + supplier SOPs |
| API | Amazon API Gateway | Secure REST endpoint |
| UI | Streamlit | Chat interface |

---

## 🗄️ DynamoDB Composite Key Design

```
Table: AutoPartsInventory
├── Partition Key (PK): part_id          e.g. PART-BRK-PAD
└── Sort Key (SK):      vehicle_model#variant   e.g. HONDA_CITY#2022

Global Secondary Indexes:
├── VehicleModelIndex   → All parts for a specific vehicle model
├── CategoryStockIndex  → All consumables sorted by stock level
└── SupplierIndex       → Parts by supplier sorted by lead time
```

**Access Patterns Supported:**
- Get exact part for exact vehicle → `GetItem(part_id, HONDA_CITY#2022)`
- Get all vehicles using a part → `Query(PK=part_id)`
- Get all low parts for Honda City → `Query(GSI-1, vehicle_model=HONDA_CITY)`
- Get all low consumables → `Query(GSI-2, category=Consumable)`

---

## 📐 Restock Formula

```
Recommended Restock = (Predicted Demand × Lead Time Weeks)
                    + Dynamic Safety Stock
                    − Current Stock

Dynamic Safety Stock = Z × demand_std × √(lead_time_weeks)
Z = 1.645  (95% service level)
```

---

## 💬 Example Queries

| Query | Response Type |
|-------|--------------|
| "Which parts are critically low right now?" | DynamoDB + ML Forecast + Bedrock |
| "Which parts are low for Honda City?" | GSI Query + Restock Recommendation |
| "How much engine oil should we reorder?" | Formula Calculation + Supplier Info |
| "What is the return policy for overstocked parts?" | RAG → Policy Document |
| "Who won the cricket match?" | Warm Out-of-Scope Redirect |
| "Hello!" | Friendly Greeting + Suggestions |

---

## 📁 Project Structure

```
auto_parts_agent/
├── 01_data_prep/
│   ├── generate_data.py          # Generate synthetic CSV datasets
│   └── upload_to_s3.py           # Upload to S3 data lake
├── 03_processing/
│   ├── preprocessing.py          # Feature engineering (runs in SageMaker)
│   └── run_processing_job.py     # Launch Processing Job
├── 04_training/
│   └── run_training_job.py       # Launch XGBoost Training Job
├── 05_registry/
│   └── register_model.py         # Register in Model Registry
├── 06_batch_transform/
│   └── run_batch_transform.py    # Run batch predictions
├── 07_dynamodb/
│   ├── create_table.py           # Create table + 3 GSIs
│   └── load_data.py              # Load 204 inventory items
├── 08_lambda/
│   └── lambda_function.py        # Lambda handler (restock + Bedrock)
├── 10_rag/
│   └── embed_documents.py        # Embed OEM docs into S3 Vectors
├── 11_ui/
│   └── app.py                    # Streamlit chat interface
├── config.py                     # Shared configuration
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas numpy boto3 sagemaker streamlit requests
```

### AWS Configuration
```bash
aws configure
# Enter: Access Key ID, Secret Access Key, Region (ap-southeast-1), json
```

### Update config.py
```python
ACCOUNT_ID      = "your_12_digit_account_id"
SAGEMAKER_ROLE  = "arn:aws:iam::YOUR_ACCOUNT:role/SageMakerExecutionRole"
REGION          = "ap-southeast-1"
S3_BUCKET       = "auto-parts-inventory"
```

### Run Step by Step

```bash
# Step 1 — Generate and upload data
python 01_data_prep/generate_data.py
python 01_data_prep/upload_to_s3.py

# Step 3 — Feature engineering (SageMaker Studio)
python 03_processing/run_processing_job.py

# Step 4 — Train XGBoost model
python 04_training/run_training_job.py

# Step 5 — Register model
python 05_registry/register_model.py

# Step 6 — Batch predictions
python 06_batch_transform/run_batch_transform.py

# Step 7 — Create DynamoDB table and load data
python 07_dynamodb/create_table.py
python 07_dynamodb/load_data.py

# Step 8 — Deploy lambda_function.py to AWS Lambda console

# Step 10 — Embed RAG documents
python 10_rag/embed_documents.py

# Step 11 — Run Streamlit UI
streamlit run 11_ui/app.py
```

---

## 💰 POC Cost Estimation

| Service | Cost |
|---------|------|
| Amazon S3 | ~$0.78/month |
| Amazon DynamoDB (on-demand) | ~$0.01/month |
| SageMaker Processing Job | ~$0.03/run |
| SageMaker Training Job | ~$0.10/run (135 seconds!) |
| SageMaker Batch Transform | ~$0.05/run |
| Amazon Bedrock Nova Lite (50 prompts) | ~$0.61/month |
| AWS Lambda | $0.00 (free tier) |
| **Total (jobs only)** | **~$5/month** |

> 💡 **Key design decision:** We use Batch Transform instead of a real-time endpoint — saving ~$0.128/hr idle cost.

---

## 📊 Results

| Metric | Value |
|--------|-------|
| Training time | 135 seconds |
| Training cost | ~$0.10 |
| DynamoDB items loaded | 204 |
| Training rows | 13,903 |
| Validation rows | 743 |
| Query types tested | 5 ✅ |
| Error rate shown to users | 0% |

---

## 🔒 AWS Sandbox SOP

- ✅ All services are pay-per-use (no idle charges)
- ✅ No real-time endpoints (use Batch Transform)
- ✅ DynamoDB on-demand capacity
- ✅ Mandatory resource tags on all resources
- ✅ All data is synthetic (no real PII)
- ✅ Cleanup after POC

---

## 🤝 Author

**Vijayasudhan G**
- GitHub: [@gvijayasudhan-boop](https://github.com/gvijayasudhan-boop)
- LinkedIn: [Add your LinkedIn URL]

---

## 📄 License

This project is for educational and demonstration purposes.

---

⭐ **If you found this project helpful, please give it a star!**
