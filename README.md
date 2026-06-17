# News Research Assistant — RAG over Live Articles

A retrieval-augmented generation (RAG) tool that answers questions over a set of news article URLs, with source attribution for every answer. Give it links, ask a question in plain English, and it returns an answer grounded in the actual article text — not the model's general knowledge — along with which article it came from.

## Why this exists

I built this while transitioning into AI/ML to get hands-on with the full RAG pipeline end-to-end: scraping unstructured web content, chunking it sensibly, embedding it, storing it in a vector index, and wiring retrieval into an LLM call. I started from a well-known LangChain tutorial pattern and then ported the entire embeddings + LLM layer from OpenAI to Google's Gemini API, which is where most of the actual engineering work happened (see below).

## How it works

1. **Ingest** — `UnstructuredURLLoader` pulls and parses raw text from a list of article URLs.
2. **Chunk** — `RecursiveCharacterTextSplitter` splits the text into 1000-character chunks with 200-character overlap, so context isn't cut off mid-thought at chunk boundaries.
3. **Embed** — Each chunk is embedded using Google's `gemini-embedding-001` model.
4. **Index** — Embeddings are stored in a local FAISS index and persisted to disk, so it doesn't need to be rebuilt on every run.
5. **Retrieve + Answer** — `RetrievalQAWithSourcesChain` retrieves the most relevant chunks for a query and asks `gemini-2.5-flash` to answer using only that retrieved context, returning the source URL alongside the answer.

```
URLs → UnstructuredURLLoader → RecursiveCharacterTextSplitter
     → Gemini Embeddings → FAISS Index (persisted locally)
     → Query → Retriever → Gemini 2.5 Flash → Answer + Source
```

## The actual engineering problem I solved

The tutorial I started from was written against OpenAI's API. Porting it to Gemini wasn't a drop-in swap — once I changed the embeddings and LLM calls to `langchain-google-genai`, the existing `langchain` / `langchain-community` versions on my machine were incompatible with that integration package, with `langchain-classic` thrown into the mix from a separate install. Imports failed, and not in a clean, well-documented way.

I resolved it by uninstalling the full LangChain stack and reinstalling a known-compatible set of pinned versions:

```
langchain==0.2.17
langchain-community==0.2.19
langchain-google-genai==1.0.10
```

This is the unglamorous but real part of working with fast-moving LLM tooling: the libraries change quickly enough that tutorial code breaks within months, and figuring out *which* package versions actually work together is its own debugging skill, separate from understanding the RAG architecture itself.

## Tech stack

- **Orchestration:** LangChain (`langchain`, `langchain-community`)
- **LLM + Embeddings:** Google Gemini (`gemini-2.5-flash`, `gemini-embedding-001`) via `langchain-google-genai`
- **Vector store:** FAISS (local, persisted to disk)
- **Document loading:** `unstructured`
- **Secrets management:** `python-dotenv` (API key loaded from `.env`, never hardcoded)

## Setup

```bash
git clone <your-repo-url>
cd <repo-name>
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
GOOGLE_API_KEY=your_api_key_here
```

## Usage

```python
from langchain_community.document_loaders import UnstructuredURLLoader

urls = [
    "https://example.com/article-1",
    "https://example.com/article-2",
]

# Load → chunk → embed → index (see notebook for full pipeline)
# Then query:
query = "What is the price of the product mentioned in the article?"
result = chain({"question": query}, return_only_outputs=True)
print(result)
```

The FAISS index is saved locally after the first build (`faiss_index/`), so subsequent runs can load it directly instead of re-embedding everything.

## What I'd improve next

- Add a Streamlit UI (imports are already in place; not yet wired up) so URLs and questions can be entered without touching the notebook.
- Clean up leftover naming from the OpenAI-based original (e.g. a FAISS index variable still named `vectorindex_openai` despite using Gemini embeddings — a small but telling artifact of the porting process).
- Add basic error handling for unreachable URLs or empty article extraction.
- Evaluate retrieval quality more rigorously (currently judged informally via spot-check queries).

## Acknowledgment

The overall pipeline structure (loaders → splitter → FAISS → retrieval chain) follows a well-known LangChain RAG tutorial pattern. The provider migration to Gemini, the dependency resolution, and the testing were done independently.
