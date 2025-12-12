# app.py - Main Streamlit app for RAG-powered documentation chatbot
import os
import json
import re
from pathlib import Path
from urllib.parse import urlparse
import streamlit as st
import requests
from bs4 import BeautifulSoup
from pinecone import Pinecone
import cohere
from openai import OpenAI
from security_utils import UrlValidationError, validate_outbound_url
from rate_limit import RateLimitExceeded, check_rate_limit

st.set_page_config(page_title="RAG Documentation Chatbot", layout="wide")

# Optional simple authentication (recommended for any non-local deployment)
APP_PASSWORD = os.getenv("APP_PASSWORD") or None
if APP_PASSWORD:
    if "auth_ok" not in st.session_state:
        st.session_state.auth_ok = False

    if not st.session_state.auth_ok:
        with st.sidebar:
            st.subheader("Access")
            provided = st.text_input("Password", type="password")
            if st.button("Unlock", use_container_width=True):
                st.session_state.auth_ok = bool(provided) and provided == APP_PASSWORD
            if not st.session_state.auth_ok:
                st.info("Enter password to use the app.")
        st.stop()

# Get absolute paths relative to this file
APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR / "data"
CRAWLED_DATA_PATH = DATA_DIR / "crawled.jsonl"

# Ensure data directory exists
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Configuration 
try:
    COHERE_API_KEY = st.secrets["COHERE_API_KEY"]
    PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    PINECONE_INDEX = st.secrets.get("PINECONE_INDEX", "website-knowledge-index")
except:
    try:
        from dotenv import load_dotenv
        load_dotenv()
        COHERE_API_KEY = os.getenv("COHERE_API_KEY")
        PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        PINECONE_INDEX = os.getenv("PINECONE_INDEX", "website-knowledge-index")
    except:
        st.error("⚠️ Missing API keys! Please configure secrets.")
        st.stop()

if not all([COHERE_API_KEY, PINECONE_API_KEY, OPENAI_API_KEY]):
    st.error("⚠️ One or more API keys are missing!")
    st.stop()

EMBED_MODEL = "embed-multilingual-v3.0"
TOP_K = 3

# Safety limits (tunable via env)
SCRAPE_LIMIT_PER_HOUR = int(os.getenv("SCRAPE_LIMIT_PER_HOUR", "30"))
QUERY_LIMIT_PER_MINUTE = int(os.getenv("QUERY_LIMIT_PER_MINUTE", "60"))

# Initialize API clients
co = cohere.Client(COHERE_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)
openai_client = OpenAI(api_key=OPENAI_API_KEY)


@st.cache_resource
def load_text_lookup():
    """Load full text lookup from crawled data file"""
    lookup = {}
    
    if CRAWLED_DATA_PATH.exists():
        try:
            with open(CRAWLED_DATA_PATH, 'r', encoding='utf-8') as f:
                for idx, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        lookup[str(idx)] = {
                            "text": record.get("text", ""),
                            "url": record.get("url", "")
                        }
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            st.sidebar.error(f"Error loading file: {e}")
    
    return lookup


# Load initial data
text_lookup = load_text_lookup()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "scraped_urls" not in st.session_state:
    # Initialize with pre-scraped URLs from file
    st.session_state.scraped_urls = set([record["url"] for record in text_lookup.values() if record.get("url")])
if "next_id" not in st.session_state:
    st.session_state.next_id = len(text_lookup)

# UI
st.title("RAG-Powered Documentation Chatbot")
st.markdown("Ask questions about indexed content or provide a URL to scrape and index!")

# Sidebar
with st.sidebar:
    st.subheader("System Status")
    st.caption(f"**Data file:** {CRAWLED_DATA_PATH}")
    st.caption(f"**File exists:** {CRAWLED_DATA_PATH.exists()}")
    st.caption(f"**Indexed pages:** {len(text_lookup)}")
    
    st.divider()
    st.subheader("Indexed Websites")
    
    if st.session_state.scraped_urls:
        domains = {}
        for url in st.session_state.scraped_urls:
            domain = urlparse(url).netloc
            domains[domain] = domains.get(domain, 0) + 1
        
        for domain, count in sorted(domains.items()):
            st.caption(f"• {domain} ({count} pages)")
    else:
        st.caption("No sites indexed yet. Provide a URL to get started!")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        if message["role"] == "assistant":
            # Show query rewriting if different
            if message.get("rewritten_query") and message.get("original_query"):
                if message["rewritten_query"] != message["original_query"]:
                    with st.expander("Query Rewriting (for better retrieval)"):
                        st.markdown(f"**Original:** {message['original_query']}")
                        st.markdown(f"**Rewritten:** {message['rewritten_query']}")
            
            # Show sources
            if message.get("sources"):
                with st.expander("Sources"):
                    for i, source in enumerate(message["sources"], 1):
                        st.markdown(f"{i}. [{source['url']}]({source['url']}) (score: {source['score']:.3f})")


def detect_url(text: str):
    """Extract URL from text using regex"""
    url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
    urls = re.findall(url_pattern, text)
    return urls[0] if urls else None


def get_domain_namespace(url: str) -> str:
    """Convert URL domain to valid Pinecone namespace"""
    domain = urlparse(url).netloc
    # Clean for namespace: alphanumeric and dash only
    namespace = re.sub(r'[^a-z0-9-]', '-', domain.lower())
    return namespace


def scrape_single_page(url: str) -> dict:
    """Scrape a single page (no recursion for memory efficiency)"""
    try:
        try:
            check_rate_limit(
                st.session_state,
                "rate:scrape",
                max_calls=SCRAPE_LIMIT_PER_HOUR,
                window_seconds=60 * 60,
            )
        except RateLimitExceeded as e:
            st.warning(str(e))
            return None

        try:
            url = validate_outbound_url(url)
        except UrlValidationError as e:
            st.warning(f"Blocked URL: {e}")
            return None

        response = requests.get(url, timeout=15, headers={
            "User-Agent": "RAGBot/1.0"
        })
        
        if response.status_code != 200:
            return None
        
        # Parse HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove unwanted elements
        for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
            tag.decompose()
        
        # Extract text
        text = soup.get_text(separator='\n')
        lines = [line.strip() for line in text.splitlines()]
        text = '\n'.join(line for line in lines if line)
        
        # Only keep if substantial content
        if len(text) < 200:
            return None
        
        return {"url": url, "text": text}
        
    except Exception as e:
        st.error(f"Scraping failed: {str(e)}")
        return None


def add_to_pinecone(page_data: dict, namespace: str):
    """Add page to Pinecone and update local storage"""
    
    # Get next ID
    next_id = str(st.session_state.next_id)
    st.session_state.next_id += 1
    
    # Embed the text
    try:
        embedding = co.embed(
            texts=[page_data["text"]],
            model=EMBED_MODEL,
            input_type="search_document",
            truncate="END"
        ).embeddings[0]
    except Exception as e:
        st.error(f"Embedding failed: {e}")
        return False
    
    # Upsert to Pinecone with namespace
    try:
        index.upsert(
            vectors=[{
                "id": next_id,
                "values": embedding,
                "metadata": {
                    "url": page_data["url"],
                    "preview": page_data["text"][:200]
                }
            }],
            namespace=namespace
        )
    except Exception as e:
        st.error(f"Pinecone upsert failed: {e}")
        return False
    
    # Update in-memory lookup
    text_lookup[next_id] = page_data
    
    # Append to file
    try:
        with open(CRAWLED_DATA_PATH, 'a', encoding='utf-8') as f:
            f.write(json.dumps(page_data, ensure_ascii=False) + '\n')
    except Exception as e:
        st.warning(f"Could not write to file (data still in Pinecone): {e}")
    
    return True


def embed_query(q: str):
    """Embed query using Cohere"""
    return co.embed(
        texts=[q],
        model=EMBED_MODEL,
        input_type="search_query",
        truncate="END",
    ).embeddings[0]


def retrieve(query_vec, k: int):
    """Retrieve from ALL namespaces in Pinecone"""
    try:
        # Get all namespaces
        stats = index.describe_index_stats()
        namespaces = list(stats.get('namespaces', {}).keys())
        
        # If no namespaces, query default (empty string)
        if not namespaces:
            namespaces = ['']
        
        # Query each namespace and combine results
        all_matches = []
        for ns in namespaces:
            try:
                result = index.query(
                    vector=query_vec,
                    top_k=k,
                    include_metadata=True,
                    namespace=ns if ns else None
                )
                all_matches.extend(result.get('matches', []))
            except Exception as e:
                continue
        
        # Sort by score and take top K
        all_matches.sort(key=lambda x: x.get('score', 0), reverse=True)
        return {"matches": all_matches[:k]}
        
    except Exception as e:
        st.error(f"Retrieval error: {str(e)}")
        return {"matches": []}


def extract_sources(matches):
    """Extract sources with full text from lookup"""
    sources = []
    for m in matches:
        meta = m.get("metadata", {}) if isinstance(m, dict) else m.metadata
        match_id = m.get("id") if isinstance(m, dict) else m.id
        score = m.get("score") if isinstance(m, dict) else m.score
        
        url = meta.get("url", "")
        
        # Get full text from lookup
        full_text = ""
        if match_id in text_lookup:
            full_text = text_lookup[match_id]["text"]
        
        sources.append({
            "url": url,
            "text": full_text,
            "score": score,
        })
    return sources


def build_context(matches) -> str:
    """Build context string from matches"""
    parts = []
    for m in matches:
        meta = m.get("metadata", {}) if isinstance(m, dict) else m.metadata
        match_id = m.get("id") if isinstance(m, dict) else m.id
        url = meta.get("url")
        preview = meta.get("preview", "")

        # Get full text from lookup
        text = ""
        if match_id in text_lookup:
            text = text_lookup[match_id]["text"]
        
        # Fallback to preview if no full text
        if not text and preview:
            text = preview

        if text and url:
            parts.append(f"Source: {url}\n{text}")
    
    return "\n\n---\n\n".join(parts)


def rewrite_query_with_context(query: str) -> str:
    """Rewrite query using chat history, stripping URLs"""
    try:
        check_rate_limit(
            st.session_state,
            "rate:rewrite",
            max_calls=QUERY_LIMIT_PER_MINUTE,
            window_seconds=60,
        )
    except RateLimitExceeded:
        query_without_url = re.sub(r'https?://[^\s]+', '', query).strip()
        return query_without_url or query
    
    # Strip URLs from query for rewriting
    query_without_url = re.sub(r'https?://[^\s]+', '', query).strip()
    
    # If no history or empty query, return as-is
    if len(st.session_state.messages) < 2 or not query_without_url:
        return query_without_url or query
    
    # Get recent chat history (last 3 pairs = 6 messages)
    recent_messages = st.session_state.messages[-6:]
    history_text = "\n".join([
        f"{msg['role']}: {msg['content']}" 
        for msg in recent_messages
    ])
    
    system_prompt = """You are a query rewriter. Given a chat history and a follow-up question, 
rewrite the question to be standalone and include necessary context from the conversation.
If the question already has full context, return it unchanged.
Only output the rewritten question, nothing else."""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Chat History:\n{history_text}\n\nQuestion: {query_without_url}"}
    ]
    
    try:
        resp = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.1,
            max_tokens=100
        )
        return resp.choices[0].message.content.strip()
    except:
        return query_without_url


def generate_with_openai(context: str, query: str) -> str:
    """Generate response using OpenAI with chat history"""
    check_rate_limit(
        st.session_state,
        "rate:answer",
        max_calls=QUERY_LIMIT_PER_MINUTE,
        window_seconds=60,
    )
    
    system_msg = {
        "role": "system",
        "content": (
            "You answer strictly using the provided sources and previous chat context."
            "If the answer cannot be found in the sources, say you do not know. "
            "Be concise, cite the source URLs inline when referencing facts."
        ),
    }
    
    messages = [system_msg]
    
    # Add recent chat history (last 5 pairs = 10 messages)
    recent_messages = st.session_state.messages[-10:]
    for msg in recent_messages:
        messages.append({
            "role": msg["role"],
            "content": msg["content"]
        })
    
    # Add current query with sources
    messages.append({
        "role": "user",
        "content": f"SOURCES:\n{context}\n\nQUESTION: {query}"
    })
    
    resp = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.2,
    )
    return resp.choices[0].message.content


# Chat input
if prompt := st.chat_input("Ask a question or provide a URL to scrape..."):
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        # Check if prompt contains a URL
        detected_url = detect_url(prompt)
        
        if detected_url:
            # URL detected - scraping mode
            if detected_url in st.session_state.scraped_urls:
                answer = f"{detected_url} is already indexed. Ask a question about it when ready."
            else:
                with st.spinner(f"Scraping {detected_url}..."):
                    page_data = scrape_single_page(detected_url)

                    if page_data:
                        namespace = get_domain_namespace(detected_url)
                        success = add_to_pinecone(page_data, namespace)

                        if success:
                            st.session_state.scraped_urls.add(detected_url)
                            answer = f"Successfully scraped and indexed {detected_url}. You can now query it."
                        else:
                            answer = f"Failed to index {detected_url}. Please try again."
                    else:
                        answer = f"Failed to scrape {detected_url}. Please check the URL and try again."
            
            st.markdown(answer)
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })
        
        else:
            # Normal question - answering mode
            with st.spinner("Thinking..."):
                try:
                    # Rewrite query with context
                    rewritten_query = rewrite_query_with_context(prompt)
                    
                    # Embed and retrieve
                    qvec = embed_query(rewritten_query)
                    res = retrieve(qvec, TOP_K)
                    matches = res["matches"]
                    sources = extract_sources(matches)
                    context = build_context(matches)
                    
                    if not context:
                        answer = "I don't have enough information to answer that question. Try providing a URL to scrape!"
                    else:
                        answer = generate_with_openai(context, prompt)
                    
                    st.markdown(answer)
                    
                    # Show query rewriting if different
                    if rewritten_query != prompt:
                        with st.expander("Query Rewriting"):
                            st.markdown(f"**Original:** {prompt}")
                            st.markdown(f"**Rewritten:** {rewritten_query}")
                    
                    # Show sources
                    if sources:
                        with st.expander("Sources"):
                            for i, source in enumerate(sources, 1):
                                st.markdown(f"{i}. [{source['url']}]({source['url']}) (score: {source['score']:.3f})")
                    
                    # Save to history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                        "original_query": prompt,
                        "rewritten_query": rewritten_query
                    })
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
