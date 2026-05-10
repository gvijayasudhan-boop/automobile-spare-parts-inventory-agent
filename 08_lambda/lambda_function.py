import json
import boto3
import os
import io
import math
from decimal import Decimal
from boto3.dynamodb.conditions import Key, Attr

# ── Clients ───────────────────────────────────────────────────────────────────
REGION           = os.environ.get("REGION",           "ap-southeast-1")
S3_BUCKET        = os.environ.get("S3_BUCKET",        "auto-parts-inventory")
DYNAMO_TABLE     = os.environ.get("DYNAMO_TABLE",     "AutoPartsInventory")
BEDROCK_MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "apac.amazon.nova-lite-v1:0")
S3_VECTORS_BUCKET= os.environ.get("S3_VECTORS_BUCKET","auto-parts-vectors")
S3_VECTORS_INDEX = os.environ.get("S3_VECTORS_INDEX", "parts-policy-index")

dynamo  = boto3.resource("dynamodb",       region_name=REGION)
table   = dynamo.Table(DYNAMO_TABLE)
s3      = boto3.client("s3",               region_name=REGION)
bedrock = boto3.client("bedrock-runtime",  region_name=REGION)
try:
    s3vectors = boto3.client("s3vectors",  region_name=REGION)
except Exception:
    s3vectors = None

# ── Keywords ──────────────────────────────────────────────────────────────────
SCOPE_KEYWORDS = [
    "part","stock","reorder","restock","inventory","spare","brake","filter",
    "oil","tyre","tire","battery","spark","plug","coolant","wiper","clutch",
    "supplier","shortage","vehicle","honda","maruti","hyundai","toyota",
    "tata","kia","forecast","demand","compatible","service","repair",
    "fitment","oem","warranty","engine","air","fuel","suspension","shock",
    "critical","low","order","quantity","lead","time","category","consumable",
]

POLICY_KEYWORDS = [
    "policy","sop","warranty","return","guideline","procedure","manual",
    "agreement","rule","standard","oem","fitment","compatible",
]

SYSTEM_PROMPT = """You are an Automobile Spare Parts Inventory Assistant for a multi-brand service center.
You help service managers and procurement teams with:
  1. Spare parts stock levels and shortage alerts
  2. Restock recommendations with clear, detailed explanations
  3. Vehicle model and variant compatibility queries
  4. Supplier and lead time information
  5. Policy questions grounded in OEM and supplier documents

Rules:
- Always explain WHY a restock is recommended, not just HOW MUCH
- When recommending a restock quantity, show the formula used
- If a query is outside your domain, respond warmly and redirect with 2-3 specific suggestions
- NEVER return a technical error message to the user
- Keep responses concise and actionable for a service manager
- Use the provided inventory data and policy context in your response
"""

Z_SCORE_95 = 1.645

# ── Helpers ───────────────────────────────────────────────────────────────────
def is_in_scope(query):
    q = query.lower()
    return any(k in q for k in SCOPE_KEYWORDS)

def is_policy_query(query):
    q = query.lower()
    return any(k in q for k in POLICY_KEYWORDS)

def decimal_to_float(obj):
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, dict):
        return {k: decimal_to_float(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [decimal_to_float(i) for i in obj]
    return obj

def build_response(text, response_type="info"):
    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
        },
        "body": json.dumps({
            "response": text,
            "type":     response_type,
        })
    }

# ── DynamoDB Queries ──────────────────────────────────────────────────────────
def get_all_critical_parts():
    results = []
    for category in ["Consumable", "Wear Item", "OEM Part"]:
        try:
            response = table.query(
                IndexName = "CategoryStockIndex",
                KeyConditionExpression = Key("category").eq(category),
                FilterExpression = Attr("shortage_flag").eq(True),
            )
            results.extend(response.get("Items", []))
        except Exception as e:
            print(f"DynamoDB query error for {category}: {e}")
    return decimal_to_float(results)

def get_parts_for_vehicle(vehicle_model):
    try:
        response = table.query(
            IndexName = "VehicleModelIndex",
            KeyConditionExpression = Key("vehicle_model").eq(vehicle_model.upper()),
            FilterExpression = Attr("shortage_flag").eq(True),
        )
        return decimal_to_float(response.get("Items", []))
    except Exception as e:
        print(f"DynamoDB vehicle query error: {e}")
        return []

def get_forecast_for_part(part_id, vehicle_model=None):
    try:
        obj    = s3.get_object(Bucket=S3_BUCKET, Key="forecast/forecast_output.csv")
        import csv
        reader = csv.DictReader(io.StringIO(obj["Body"].read().decode()))
        for row in reader:
            if row.get("part_id") == part_id:
                if vehicle_model and row.get("vehicle_model","").upper() != vehicle_model.upper():
                    continue
                return {
                    "part_id":          row["part_id"],
                    "vehicle_model":    row.get("vehicle_model",""),
                    "predicted_demand": float(row.get("predicted_demand", 0)),
                }
        return {}
    except Exception as e:
        print(f"S3 forecast read error: {e}")
        return {}

# ── Restock Formula ───────────────────────────────────────────────────────────
def calculate_restock(item, forecast):
    current_stock  = float(item.get("current_stock",  0))
    reorder_level  = float(item.get("reorder_level",  0))
    lead_time_days = float(item.get("lead_time_days", 7))
    moq            = float(item.get("moq",            1))
    pred_demand    = float(forecast.get("predicted_demand", reorder_level * 2))

    lead_weeks   = lead_time_days / 7.0
    demand_std   = pred_demand * 0.25
    safety_stock = Z_SCORE_95 * demand_std * math.sqrt(lead_weeks)
    raw_restock  = (pred_demand * lead_weeks) + safety_stock - current_stock
    restock_qty  = max(0.0, raw_restock)

    if moq > 0 and restock_qty > 0:
        restock_qty = math.ceil(restock_qty / moq) * moq

    is_critical = current_stock <= reorder_level
    is_single   = bool(item.get("single_source", False))

    return {
        "part_id":            item.get("part_id"),
        "part_name":          item.get("part_name", ""),
        "vehicle_model":      item.get("vehicle_model", ""),
        "variant":            item.get("sk","").split("#")[-1] if "#" in item.get("sk","") else "",
        "current_stock":      int(current_stock),
        "reorder_level":      int(reorder_level),
        "predicted_demand":   round(pred_demand, 1),
        "safety_stock":       round(safety_stock, 1),
        "restock_qty":        int(restock_qty),
        "supplier_id":        item.get("supplier_id",""),
        "lead_time_days":     int(lead_time_days),
        "moq":                int(moq),
        "unit_cost":          item.get("unit_cost", 0),
        "is_critical":        is_critical,
        "single_source_risk": is_single,
        "status":             "CRITICAL" if is_critical else "LOW" if current_stock <= reorder_level * 1.5 else "OK",
    }

# ── RAG Retrieval ─────────────────────────────────────────────────────────────
def rag_retrieve(query, top_k=3):
    if s3vectors is None:
        return ""
    try:
        embed_resp = bedrock.invoke_model(
            modelId     = "amazon.titan-embed-text-v1",
            contentType = "application/json",
            accept      = "application/json",
            body        = json.dumps({"inputText": query}),
        )
        query_vector = json.loads(embed_resp["body"].read())["embedding"]
        search_resp  = s3vectors.search_vectors(
            VectorBucketName = S3_VECTORS_BUCKET,
            IndexName        = S3_VECTORS_INDEX,
            QueryVector      = query_vector,
            TopK             = top_k,
            ReturnMetadata   = True,
        )
        chunks = [
            hit.get("Metadata", {}).get("text", "")
            for hit in search_resp.get("SearchResults", [])
        ]
        return "\n\n".join(c for c in chunks if c)
    except Exception as e:
        print(f"RAG retrieval failed: {e}")
        return ""

# ── Bedrock Invocation (FIXED for APAC Nova Lite) ─────────────────────────────
def invoke_bedrock(user_message, system_prompt=None, context=""):
    """Call Amazon Bedrock — correct format for APAC Nova Lite inference profile."""
    if system_prompt is None:
        system_prompt = SYSTEM_PROMPT

    full_message = user_message
    if context:
        full_message = f"Context:\n{context}\n\nUser query: {user_message}"

    # FIXED: correct request body format for Nova Lite APAC profile
    body = {
        "messages": [
            {
                "role": "user",
                "content": [{"text": full_message}]   # content must be a list
            }
        ],
        "system": [{"text": system_prompt}],           # system must be a list
        "inferenceConfig": {                           # maxTokens not max_tokens
            "maxTokens":   600,
            "temperature": 0.2,
        },
    }

    response = bedrock.invoke_model(
        modelId     = BEDROCK_MODEL_ID,
        contentType = "application/json",
        accept      = "application/json",
        body        = json.dumps(body),
    )
    result = json.loads(response["body"].read())

    # Parse Nova Lite response format
    try:
        return result["output"]["message"]["content"][0]["text"]
    except Exception:
        # Fallback for other model response formats
        content = result.get("content", [])
        if isinstance(content, list) and content:
            return content[0].get("text", "")
        return str(result)

# ── Out-of-Scope Handler ──────────────────────────────────────────────────────
def handle_out_of_scope(query):
    out_of_scope_system = """You are an Automobile Spare Parts Inventory Assistant.
The user has asked something outside your domain.
Respond warmly. Acknowledge briefly. Then redirect by suggesting 2-3 specific things they can ask you.
Examples of things you CAN help with:
  - Which spare parts are running low for Honda City this week?
  - How much engine oil should we reorder?
  - What brake pads are compatible with Maruti Swift 2021?
Never say 'error'. Never say 'I dont know'. Always be helpful and friendly."""

    text = invoke_bedrock(query, system_prompt=out_of_scope_system)
    return build_response(text, "redirect")

# ── Parts Query Handler ───────────────────────────────────────────────────────
def handle_parts_query(query):
    # 1. RAG if policy question
    rag_context = ""
    if is_policy_query(query):
        rag_context = rag_retrieve(query)

    # 2. Detect vehicle model
    detected_vehicle = None
    for vm in ["MARUTI_SWIFT","HONDA_CITY","HYUNDAI_I20","TOYOTA_INNOVA","TATA_NEXON","KIA_SELTOS"]:
        if vm.replace("_"," ").lower() in query.lower() or vm.lower() in query.lower():
            detected_vehicle = vm
            break

    # 3. Fetch inventory
    restock_results = []
    if detected_vehicle:
        parts = get_parts_for_vehicle(detected_vehicle)
    else:
        parts = get_all_critical_parts()

    # 4. Calculate restock
    for item in parts[:10]:
        forecast = get_forecast_for_part(item.get("part_id",""), detected_vehicle)
        rs = calculate_restock(item, forecast)
        restock_results.append(rs)

    # 5. Build inventory summary
    inventory_summary = ""
    if restock_results:
        lines = []
        for rs in sorted(restock_results, key=lambda x: (0 if x["is_critical"] else 1)):
            flag = "CRITICAL" if rs["is_critical"] else "LOW"
            lines.append(
                f"{flag} | {rs['part_name']} ({rs['part_id']}) | "
                f"Model: {rs['vehicle_model']} | "
                f"Stock: {rs['current_stock']} / Reorder: {rs['reorder_level']} | "
                f"Restock: {rs['restock_qty']} units | "
                f"Predicted demand: {rs['predicted_demand']} | "
                f"Supplier: {rs['supplier_id']} | Lead: {rs['lead_time_days']} days"
                + (" | SINGLE SOURCE RISK" if rs["single_source_risk"] else "")
            )
        inventory_summary = "Current inventory status:\n" + "\n".join(lines)

    # 6. Build context
    context = inventory_summary
    if rag_context:
        context += f"\n\nRelevant policy/OEM documentation:\n{rag_context}"

    # 7. Call Bedrock
    response_text = invoke_bedrock(query, context=context)
    return build_response(response_text, "restock" if restock_results else "policy")

# ── Lambda Handler ────────────────────────────────────────────────────────────
def lambda_handler(event, context):
    try:
        # Handle API Gateway and direct invocation
        if "body" in event:
            body = json.loads(event["body"]) if isinstance(event["body"], str) else event["body"]
        else:
            body = event

        query = body.get("query", "").strip()

        if not query:
            return build_response(
                "Please ask me about spare parts availability, restock recommendations, "
                "or vehicle compatibility. For example: Which parts are critically low this week?",
                "prompt"
            )

        if is_in_scope(query):
            return handle_parts_query(query)
        else:
            return handle_out_of_scope(query)

    except Exception as e:
        print(f"Unexpected error: {e}")
        return build_response(
            "I encountered an issue processing your request. "
            "Please try rephrasing your query about spare parts or inventory.",
            "error"
        )
