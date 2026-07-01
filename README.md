# PariShiksha-NCERT-RAG 📚

![PariShiksha Banner](https://via.placeholder.com/1200x400.png?text=PariShiksha+NCERT+RAG)

**PariShiksha-NCERT-RAG** is an advanced Retrieval-Augmented Generation (RAG) system built to serve as an intelligent, conversational tutor based strictly on NCERT Science textbooks. It combines state-of-the-art vector search with powerful LLMs (Large Language Models) to provide accurate, citation-grounded answers to student queries.

---

## ✨ Features
- **Intelligent Q&A**: Ask complex questions based on NCERT chapters, and get accurate answers.
- **Citation Grounded**: Every answer includes the exact `[Page X]` source from the textbook.
- **Hybrid Retrieval System**: Uses both Dense Vector Embeddings (ChromaDB) and Sparse Retrieval (BM25) with RRF (Reciprocal Rank Fusion) ranking.
- **Advanced Reranking**: Reorders retrieved chunks for optimal relevance before generation.
- **Robust UI Error Handling**: Sleek, dark-mode Gradio interface that initializes gracefully even when API keys are not yet configured.
- **Flexible Ragas Evaluation**: Runs automated Q&A checks falling back to Groq if Anthropic keys are missing.

---

## 🏗️ Architecture & Flow Chart

Here is how information flows through the PariShiksha retrieval and generation pipeline:

```mermaid
graph TD
    A[User Query] --> B[EnhancedRetriever]
    B --> C{use_multiquery?}
    C -- Yes --> D[Query Expansion via LLM]
    C -- No --> E[Original Query]
    D --> F[Hybrid Search]
    E --> F
    F --> G[Sparse Index: BM25]
    F --> H[Dense Index: ChromaDB]
    G --> I[RRF Fusion Ranks]
    H --> I
    I --> J[Filtered RRF Results]
    J --> K[Rerank Pipeline: MS-MARCO / Cohere]
    K --> L[Top K Context Chunks]
    L --> M[Strict Grounding Prompt]
    M --> N[LLM Generation: Llama-3.3]
    N --> O[Answer + Sources]
```

1. **Ingestion (`src/ingestion`)**: PyMuPDF extracts raw text from PDF chapters.
2. **Chunking (`src/retrieval/chunking.py`)**: Content is chunked logically with overlaps using `langchain_text_splitters`.
3. **Indexing (`src/retrieval/indexing.py`)**: Data is vectorized using HuggingFace embeddings and stored in ChromaDB and BM25 indexes.
4. **Retrieval (`src/retrieval/retrieval.py`)**: An `EnhancedRetriever` handles hybrid search, multi-query expansion, and re-ranking.
5. **Generation (`src/generation`)**: Groq (Llama-3) or Anthropic models generate grounded answers.

---

## 📁 Folder Structure
```text
PariShiksha-NCERT-RAG/
├── src/
│   ├── ingestion/       # PDF parsing logic
│   ├── retrieval/       # Chunking, BM25, and ChromaDB indexing/retrieval
│   ├── generation/      # LLM Generation pipeline
│   ├── evaluation/      # Ragas-based QA evaluation
│   └── app/             # Gradio web interface
├── data/
│   ├── raw/             # Place your raw NCERT PDFs here
│   └── processed/       # Extracted JSON chunks (chunks.json)
├── vectorstore/         # ChromaDB persistence directory (chroma_db/)
├── notebooks/           # Jupyter notebooks for experimentation
├── legacy/              # Previous versions (V1 and V2)
├── main.py              # CLI Entry point
├── requirements.txt     # Dependency list
└── .env.example         # Template for environment variables
```

---

## 🚀 Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/PariShiksha-NCERT-RAG.git
   cd PariShiksha-NCERT-RAG
   ```

2. **Create a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Variables**:
   Copy the example `.env` file and add your API keys.
   ```bash
   cp .env.example .env
   # Edit .env and insert your GROQ_API_KEY, COHERE_API_KEY, etc.
   ```

---

## 📖 How to Run

### 1. Ingestion & Indexing
Place your NCERT PDFs (e.g., `iesc107.pdf`) inside `data/raw/`.
Run the indexing pipeline to parse, chunk, and embed the text into ChromaDB and BM25:
```bash
python src/retrieval/indexing.py
```
*(Note: To re-embed and overwrite the database on changes, the script automatically triggers a `force_rebuild` flag).*

### 2. Run the Gradio App
To launch the beautiful web interface:
```bash
python main.py --app
# Or run directly: python src/app/gradio_app.py
```

### 3. Run the CLI QA
If you prefer a terminal-based Q&A loop:
```bash
python main.py --cli
```

### 4. Run Evaluation
To run RAGAS automated evaluations against the Golden Set:
```bash
python src/evaluation/evaluation.py
```

---

## 🔮 Future Improvements
- [ ] Add conversation memory / chat history across sessions.
- [ ] Incorporate OCR for parsing complex tables and equations.
- [ ] Add a feedback mechanism (thumbs up/down) in the UI for reinforcement learning.

---
*Created with ❤️ for Educational Excellence.*
