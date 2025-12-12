import time


class RateLimitExceeded(Exception):
    pass


def check_rate_limit(
    state: dict,
    key: str,
    *,
    max_calls: int,
    window_seconds: int,
    now: float | None = None,
) -> None:
    """
    Minimal in-memory (per-process) sliding-window limiter.
    Intended for Streamlit `st.session_state` or similar dict-like state.
    """
    if now is None:
        now = time.time()

    bucket = state.get(key)
    if not isinstance(bucket, list):
        bucket = []

    cutoff = now - window_seconds
    bucket = [t for t in bucket if isinstance(t, (int, float)) and t >= cutoff]

    if len(bucket) >= max_calls:
        raise RateLimitExceeded(f"Rate limit exceeded ({max_calls}/{window_seconds}s)")

    bucket.append(now)
    state[key] = bucket
