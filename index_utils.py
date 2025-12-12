# index_utils.py
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence, Tuple


Batch = Tuple[List[Dict[str, Any]], int]


def batchify(items: Sequence[Dict[str, Any]], size: int) -> Iterable[Batch]:
    if size <= 0:
        raise ValueError("batch size must be positive")
    total = len(items)
    for start in range(0, total, size):
        yield list(items[start : start + size]), start


def _resolve_metadata(record: Dict[str, Any]) -> Dict[str, Any]:
    if "metadata" in record and isinstance(record["metadata"], dict):
        return record["metadata"]
    # Fall back to all top-level keys except id/text
    meta = {k: v for k, v in record.items() if k not in {"id", "text", "metadata"}}
    meta.setdefault("text", record.get("text", ""))
    return meta


def embed_and_upsert(
    co_client,
    pinecone_index,
    records: Sequence[Dict[str, Any]],
    model: str,
    batch_size: int = 96,
    input_type: str = "search_document",
    truncate: str = "END",
    verbose: bool = True,
) -> int:
    """
    Embed records with Cohere and upsert them into the Pinecone index.

    Each record must contain a ``text`` field and may optionally provide
    ``id`` and ``metadata`` fields. When absent, IDs default to the running
    position in the sequence and metadata is derived from the record itself.
    """
    total = len(records)
    if total == 0:
        return 0

    for batch, start_idx in batchify(records, batch_size):
        texts = [rec.get("text", "") for rec in batch]
        embeddings = co_client.embed(
            texts=texts,
            model=model,
            input_type=input_type,
            truncate=truncate,
        ).embeddings

        vectors = []
        for offset, rec in enumerate(batch):
            vector_id = str(rec.get("id", start_idx + offset))
            metadata = _resolve_metadata(rec)
            vectors.append(
                {
                    "id": vector_id,
                    "values": embeddings[offset],
                    "metadata": metadata,
                }
            )

        pinecone_index.upsert(vectors=vectors)
        if verbose:
            done = min(start_idx + len(batch), total)
            print(f"Upserted {done}/{total}")

    return total
