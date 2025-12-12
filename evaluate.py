# evaluate.py
import json
import os
import time
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv
from pinecone import Pinecone
import cohere

load_dotenv()

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "website-knowledge-index")
EMBED_MODEL = os.getenv("COHERE_EMBED_MODEL", "embed-multilingual-v3.0")
EVAL_DATA_PATH = Path(os.getenv("EVAL_DATA_PATH", "data/eval_set.jsonl"))
RESULTS_PATH = Path(os.getenv("EVAL_RESULTS_PATH", "data/eval_results.json"))

assert COHERE_API_KEY and PINECONE_API_KEY

co = cohere.Client(COHERE_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)


def load_eval_items(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    if not path.exists():
        raise FileNotFoundError(f"Evaluation set not found at {path}")
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not items:
        raise ValueError(f"No evaluation records loaded from {path}")
    return items


def embed_query(q: str) -> List[float]:
    return co.embed(
        texts=[q],
        model=EMBED_MODEL,
        input_type="search_query",
        truncate="END",
    ).embeddings[0]


def contains_keywords(text: str, keywords: List[str]) -> bool:
    lower = text.lower()
    return all(kw.lower() in lower for kw in keywords)


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    if not vec_a or not vec_b:
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = sum(a * a for a in vec_a) ** 0.5
    norm_b = sum(b * b for b in vec_b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def run():
    eval_items = load_eval_items(EVAL_DATA_PATH)
    latencies = []
    topk_nonempty = 0
    keyword_hits = 0
    url_matches = 0
    top_scores: List[float] = []
    similarity_scores: List[float] = []
    results: List[Dict[str, Any]] = []

    for item in eval_items:
        q = item["question"]
        keywords = item.get("expected_keywords", [])
        target_url = item.get("reference_url", "")
        reference_answer = item.get("reference_answer")
        t0 = time.time()
        vec = embed_query(q)
        res = index.query(vector=vec, top_k=5, include_metadata=True)
        dt = (time.time() - t0) * 1000.0
        latencies.append(dt)
        matches = res.get("matches", [])
        has_text = any((m.get("metadata", {}) or {}).get("text") for m in matches)
        if has_text:
            topk_nonempty += 1
        if matches:
            best_meta = matches[0].get("metadata", {}) or {}
            best_url = best_meta.get("url") or best_meta.get("source") or "n/a"
        else:
            best_meta = {}
            best_url = "n/a"
        best_score = matches[0].get("score") if matches else None
        if best_score is not None:
            top_scores.append(best_score)

        found_keywords = False
        for m in matches:
            meta = m.get("metadata", {}) or {}
            text = meta.get("text", "") or ""
            if keywords and text and contains_keywords(text, keywords):
                found_keywords = True
                break
        if found_keywords:
            keyword_hits += 1

        if target_url and best_url != "n/a" and best_url.startswith(target_url):
            url_matches += 1

        similarity = None
        if reference_answer and best_meta.get("text"):
            embeddings = co.embed(
                texts=[reference_answer, best_meta.get("text", "")],
                model=EMBED_MODEL,
                input_type="search_document",
                truncate="END",
            ).embeddings
            similarity = cosine_similarity(embeddings[0], embeddings[1])
            similarity_scores.append(similarity)

        results.append(
            {
                "question": q,
                "latency_ms": dt,
                "top_1_url": best_url,
                "top_1_score": best_score,
                "keyword_hit": found_keywords,
                "url_match": bool(target_url and best_url != "n/a" and best_url.startswith(target_url)),
                "reference": target_url,
                "metadata": best_meta,
                "reference_similarity": similarity,
            }
        )

        print(f"Q: {q}")
        print(f"Top-1 reference: {best_url}")
        if best_score is not None:
            print(f"Top-1 score: {best_score:.4f}")
        print(f"Latency: {dt:.1f} ms")
        print(f"Keyword coverage: {'yes' if found_keywords else 'no'}")
        if similarity is not None:
            print(f"Reference similarity: {similarity:.4f}")
        print("-" * 60)

    avg_latency = sum(latencies) / max(1, len(latencies))
    coverage = 100.0 * topk_nonempty / max(1, len(eval_items))
    keyword_recall = 100.0 * keyword_hits / max(1, len(eval_items))
    avg_relevance = sum(top_scores) / max(1, len(top_scores)) if top_scores else 0.0
    url_match_rate = 100.0 * url_matches / max(1, len(eval_items))
    avg_similarity = sum(similarity_scores) / max(1, len(similarity_scores)) if similarity_scores else 0.0

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w", encoding="utf-8") as fp:
        json.dump(
            {
                "average_latency_ms": avg_latency,
                "retrieval_coverage_percent": coverage,
                "keyword_recall_percent": keyword_recall,
                "average_top1_relevance": avg_relevance,
                "url_match_percent": url_match_rate,
                "average_reference_similarity": avg_similarity,
                "samples": results,
            },
            fp,
            indent=2,
        )

    print(f"Average latency: {avg_latency:.1f} ms")
    print(f"Top-k coverage with non-empty context: {coverage:.1f}%")
    print(f"Keyword recall: {keyword_recall:.1f}%")
    print(f"Top-1 URL match rate: {url_match_rate:.1f}%")
    if top_scores:
        print(f"Average top-1 relevance score: {avg_relevance:.4f}")
    if similarity_scores:
        print(f"Average reference similarity: {avg_similarity:.4f}")
    print(f"Detailed results saved to {RESULTS_PATH}")

if __name__ == "__main__":
    run()
