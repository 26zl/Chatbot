# build_index.py
import os
from typing import List, Dict
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import cohere
from crawler import load_records
from index_utils import embed_and_upsert

load_dotenv()

# Environment
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "website-knowledge-index")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
EMBED_MODEL = os.getenv("COHERE_EMBED_MODEL", "embed-multilingual-v3.0")
BATCH = int(os.getenv("EMBED_BATCH_SIZE", "96"))
DIMENSION = 1024  # Cohere multilingual v3.0 dimensionality

assert PINECONE_API_KEY, "Missing PINECONE_API_KEY"
assert COHERE_API_KEY, "Missing COHERE_API_KEY"

co = cohere.Client(COHERE_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create index if needed (serverless)
existing = {ix["name"] for ix in pc.list_indexes()}
if PINECONE_INDEX not in existing:
    pc.create_index(
        name=PINECONE_INDEX,
        dimension=DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
index = pc.Index(PINECONE_INDEX)


def build() -> None:
    raw_records: List[Dict] = load_records()
    records = []
    for idx, rec in enumerate(raw_records):
        text = rec.get("text", "")
        url = rec.get("url", "")
        records.append(
            {
                "id": str(idx),
                "text": text,  # This gets embedded
                "metadata": {
                    "url": url,  # Only store URL, not full text!
                    "preview": text[:200]  # Optional: small preview
                },
            }
        )

    total = embed_and_upsert(
        co_client=co,
        pinecone_index=index,
        records=records,
        model=EMBED_MODEL,
        batch_size=BATCH,
        input_type="search_document",
    )
    print(f"Indexed {total} records")


if __name__ == "__main__":
    build()
