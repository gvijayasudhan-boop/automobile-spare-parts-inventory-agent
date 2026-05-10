# ─────────────────────────────────────────────────────────────────────────────
# 10_rag/embed_documents.py
# Run once to create S3 Vectors index and embed all policy/OEM documents
# Requires: documents uploaded to s3://auto-parts-inventory/documents/
# ─────────────────────────────────────────────────────────────────────────────

import boto3
import json
import os
import sys
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import S3_BUCKET, S3_VECTORS_BUCKET, S3_VECTORS_INDEX, REGION, BEDROCK_MODEL_ID

s3      = boto3.client("s3",              region_name=REGION)
bedrock = boto3.client("bedrock-runtime", region_name=REGION)

try:
    s3v = boto3.client("s3vectors", region_name=REGION)
except Exception as e:
    print(f"S3 Vectors client not available: {e}")
    s3v = None

CHUNK_SIZE = 400   # characters per chunk
OVERLAP    = 80    # overlap between chunks

# ── Sample policy documents (used if no real PDFs are available) ──────────
SAMPLE_DOCS = [
    {
        "id": "oem-fitment-brake-pads",
        "source": "OEM Fitment Guide — Brake Pads",
        "type": "fitment",
        "text": """
OEM Brake Pad Fitment Guide:
- PART-BRK-PAD-F (Front Brake Pad) is compatible with: HONDA_CITY 2020/2022, HYUNDAI_I20 2021/2023, TATA_NEXON 2023.
- PART-BRK-PAD-R (Rear Brake Pad) is compatible with: HONDA_CITY 2020/2022, HYUNDAI_I20 2021/2023, TATA_NEXON 2023.
- For MARUTI_SWIFT 2021/2023: use PART-BRK-PAD-F-SWIFT (smaller caliper size 247mm).
- For TOYOTA_INNOVA 2020/2022: use PART-BRK-PAD-F-INNOVA (larger caliper, 280mm disc).
- Brake pads should be replaced every 40,000 km or when pad thickness < 3mm.
- Always replace front and rear pads as a complete set (both sides).
- OEM warranty: 12 months or 20,000 km from date of installation.
"""
    },
    {
        "id": "safety-stock-sop",
        "source": "Workshop SOP — Safety Stock Guidelines",
        "type": "sop",
        "text": """
Safety Stock Guidelines SOP (Revision 3.1):
- Consumables (Engine Oil, Filters, Spark Plugs): Maintain minimum 30 days of safety stock.
- Wear Items (Brake Pads, Tyres, Clutch Plates): Maintain minimum 45 days of safety stock.
- OEM Parts (Timing Chain, Suspension Arms): Maintain minimum 60 days of safety stock.
- Safety stock formula: Z × standard_deviation_of_demand × sqrt(lead_time_in_weeks).
- Z-score for 95% service level = 1.645.
- Review safety stock levels quarterly or after any supplier lead time change.
- Single-source parts must maintain 2× standard safety stock levels.
- Any part with stock below reorder_level must be flagged as CRITICAL and procurement raised within 24 hours.
"""
    },
    {
        "id": "supplier-return-policy",
        "source": "Supplier Agreement — Return and Warranty Policy",
        "type": "supplier",
        "text": """
Supplier Return and Warranty Policy (Effective FY2024):
- OEM parts purchased from authorized suppliers carry a 12-month warranty from date of purchase.
- Defective parts must be returned within 30 days of identification with the original invoice.
- Engine oil and consumables: no return accepted after the seal is broken.
- Tyres: warranty covers manufacturing defects only, not wear and tear.
- Overstocked items may be returned within 90 days with a 10% restocking fee.
- Single-source supplier parts: escalate to procurement manager if lead time exceeds 14 days.
- MOQ (Minimum Order Quantity) must be respected for all supplier orders.
- Emergency orders below MOQ incur a 15% surcharge.
"""
    },
    {
        "id": "monsoon-procurement-sop",
        "source": "Seasonal Procurement SOP — Monsoon Season",
        "type": "sop",
        "text": """
Monsoon Season Procurement SOP (June to September):
- Increase safety stock for Brake Pads by 60% from May onwards.
- Increase safety stock for Tyres by 70% from May onwards.
- Increase stock of Windshield Fluid and Wiper Blades by 50%.
- Suspension Arms and Shock Absorbers: increase safety stock by 40% (poor road conditions).
- Review and confirm supplier delivery schedules before June 1st.
- Keep a buffer of 2 weeks additional stock for all wear items during June-September.
- If any critical part hits reorder level during monsoon, place emergency order immediately.
"""
    },
    {
        "id": "oil-filter-fitment",
        "source": "OEM Fitment Guide — Filters and Oils",
        "type": "fitment",
        "text": """
Filter and Oil Fitment Guide:
- Engine Oil 5W-30: Suitable for MARUTI_SWIFT, HONDA_CITY, HYUNDAI_I20, TATA_NEXON, KIA_SELTOS.
- Engine Oil 10W-40: Suitable for TOYOTA_INNOVA (diesel), older petrol engines above 100,000 km.
- Air Filter PART-AIR-FLT: Universal fit for all models. Replace every 20,000 km or 12 months.
- Oil Filter PART-OIL-FLT: Universal fit. Replace at every oil change (every 5,000-10,000 km).
- Fuel Filter PART-FUELFLTR: Replace every 40,000 km for petrol; every 20,000 km for diesel.
- Always use OEM-grade oil filters to maintain manufacturer warranty validity.
- Using non-OEM oil filters may void the engine warranty.
"""
    },
]

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> list:
    """Split text into overlapping chunks."""
    text   = " ".join(text.split())  # normalize whitespace
    chunks = []
    start  = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def get_embedding(text: str) -> list:
    """Get embedding from Amazon Bedrock Titan Embed."""
    response = bedrock.invoke_model(
        modelId     = "amazon.titan-embed-text-v1",
        contentType = "application/json",
        accept      = "application/json",
        body        = json.dumps({"inputText": text[:8000]}),  # Titan limit
    )
    return json.loads(response["body"].read())["embedding"]

def create_vector_bucket_and_index():
    """Create S3 Vectors bucket and index if they don't exist."""
    if s3v is None:
        print("S3 Vectors not available — skipping index creation.")
        return False

    try:
        s3v.create_vector_bucket(VectorBucketName=S3_VECTORS_BUCKET)
        print(f"Created S3 Vectors bucket: {S3_VECTORS_BUCKET}")
    except Exception:
        print(f"S3 Vectors bucket already exists: {S3_VECTORS_BUCKET}")

    try:
        s3v.create_index(
            VectorBucketName = S3_VECTORS_BUCKET,
            IndexName        = S3_VECTORS_INDEX,
            DataType         = "float32",
            Dimension        = 1536,           # Titan embed dimension
            DistanceMetric   = "cosine",
        )
        print(f"Created S3 Vectors index: {S3_VECTORS_INDEX}")
    except Exception:
        print(f"S3 Vectors index already exists: {S3_VECTORS_INDEX}")

    return True

def embed_and_store(docs: list):
    """Embed each document chunk and store in S3 Vectors."""
    total_chunks = 0
    for doc in docs:
        print(f"Embedding: {doc['source']}...")
        chunks = chunk_text(doc["text"])
        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                continue
            vec_id    = f"{doc['id']}-chunk-{i}"
            embedding = get_embedding(chunk)

            if s3v:
                s3v.put_vector(
                    VectorBucketName = S3_VECTORS_BUCKET,
                    IndexName        = S3_VECTORS_INDEX,
                    Key              = vec_id,
                    Vector           = embedding,
                    Metadata         = {
                        "source": doc["source"],
                        "type":   doc["type"],
                        "text":   chunk,
                        "doc_id": doc["id"],
                    }
                )
            total_chunks += 1
            print(f"  Stored chunk {i+1}/{len(chunks)}: {vec_id}")

    print(f"\nEmbedding complete. Total chunks stored: {total_chunks}")

if __name__ == "__main__":
    print("Starting RAG embedding pipeline...")
    index_ready = create_vector_bucket_and_index()

    if index_ready:
        embed_and_store(SAMPLE_DOCS)
        print(f"\nAll documents embedded in S3 Vectors.")
        print(f"  Bucket : {S3_VECTORS_BUCKET}")
        print(f"  Index  : {S3_VECTORS_INDEX}")
        print("Lambda will now retrieve relevant chunks at query time.")
    else:
        print("\nS3 Vectors not available in this region.")
        print("The Lambda function will skip RAG retrieval gracefully.")
