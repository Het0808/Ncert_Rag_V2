import os
import re
import json
import logging
import pathlib
from langchain_groq import ChatGroq
from src.retrieval.indexing import load_indexes
from src.retrieval.retrieval import EnhancedRetriever, HybridRetriever, RerankPipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

STRICT_GROUNDING_PROMPT = """You are PariShiksha, an expert NCERT Science Tutor for Grade 9.

CONTEXT FROM STUDY MATERIALS:
{context}

USER QUESTION: {question}

INSTRUCTIONS:
1. Search the CONTEXT above for the answer first.
2. If found, answer strictly using the context and cite the chunk ID: [Source: chunk_id]
3. If the answer is NOT in the context but is a valid NCERT Grade 9 Science topic (like Newton's Laws, Gravitation, etc.), provide a clear explanation using your expert knowledge of the NCERT syllabus. In this case, do NOT output any source citation, do NOT use brackets `[]`, and do NOT include the word "Source".
4. CRITICAL RULE: Under NO circumstances should your answer include phrases like "not explicitly mentioned in the provided context", "not in the context", "as an expert NCERT Science Tutor", or any other meta-commentary about the context. Do not mention that the context is missing or that you are using external knowledge. Simply provide the scientific answer directly to the user.
5. If the question is entirely outside the scope of NCERT Grade 9 Science, politely decline.

ANSWER:"""

PERMISSIVE_PROMPT = """Answer the question using the context below.

Context: {context}

Question: {question}

Answer:"""

def format_context(chunks: list[dict]) -> str:
    """Formats retrieved chunks into a context string."""
    formatted = []
    for c in chunks:
        formatted.append(f"[{c['chunk_id']}]\n{c['content']}")
    return "\n---\n".join(formatted)

def ask(question: str, retriever, temperature: float = 0, max_chunks: int = 5, use_strict: bool = True) -> dict:
    """Generates an answer using the retriever and LLM."""
    chunks = retriever.retrieve(question, k=max_chunks)
    
    if not chunks:
        return {
            "answer": "I couldn't find relevant information.",
            "sources": [],
            "chunk_ids": [],
            "chunks_used": []
        }
    
    context_str = format_context(chunks)
    template = STRICT_GROUNDING_PROMPT if use_strict else PERMISSIVE_PROMPT
    prompt_string = template.format(context=context_str, question=question)
    
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or not api_key.strip() or api_key.startswith("your_"):
        return {
            "answer": "⚠️ **GROQ_API_KEY is not set or is invalid.**\n\nPlease add your Groq API Key to the `.env` file in the project directory:\n`GROQ_API_KEY=gsk_...`\n\nOnce set, restart the application to enable LLM-based answering.",
            "sources": [],
            "chunk_ids": [],
            "chunks_used": chunks
        }
        
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=temperature)
    response = llm.invoke(prompt_string)
    answer_text = response.content
    
    # Extract chunk_ids using a more flexible regex
    cited_ids = re.findall(r'\[(?:Source:\s*)?([^\]]+)\]', answer_text)
    
    # Build sources list
    sources = []
    chunk_map = {c['chunk_id']: c for c in chunks}
    for cid in cited_ids:
        if cid in chunk_map:
            c = chunk_map[cid]
            sources.append(f"{c['metadata']['source']}, Page {c['metadata']['page']}")
            
    # Deduplicate sources
    sources = list(dict.fromkeys(sources))
    
    logger.info(f"Q: '{question[:50]}...' | Chunks: {len(chunks)} | Citations: {len(cited_ids)}")
    
    return {
        "answer": answer_text,
        "sources": sources,
        "chunk_ids": cited_ids,
        "chunks_used": chunks
    }

def compare_prompts(question: str, retriever) -> dict:
    """Compares strict and permissive grounding for a question."""
    strict_res = ask(question, retriever, use_strict=True)
    permissive_res = ask(question, retriever, use_strict=False)
    
    return {
        "question": question,
        "strict_answer": strict_res["answer"],
        "permissive_answer": permissive_res["answer"],
        "strict_citations": strict_res["chunk_ids"],
        "permissive_citations": permissive_res["chunk_ids"]
    }

if __name__ == "__main__":
    chunks_path = "./data/processed/chunks.json"
    persist_dir = "./vectorstore/chroma_db"
    collection_name = "parishiksha_v2"
    
    with open(chunks_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
        
    # Load indexes
    vs, bm25 = load_indexes(persist_dir, collection_name, chunks)
    
    # Build retriever chain
    hybrid = HybridRetriever(vs, bm25, chunks)
    reranker = RerankPipeline(use_cohere=True)
    retriever = EnhancedRetriever(hybrid, reranker, use_multiquery=True)
    
    questions = [
        "What is Newton's second law of motion?",
        "Why do all objects fall at the same rate?",
        "Who won the 2024 Nobel Prize in Physics?"
    ]
    
    results = []
    for q in questions:
        results.append(compare_prompts(q, retriever))
        
    # Save to outputs/prompt_diff.md
    output_dir = pathlib.Path("./outputs")
    output_dir.mkdir(exist_ok=True)
    
    md_content = "# Prompt Comparison Results\n\n"
    for res in results:
        md_content += f"## Question: {res['question']}\n\n"
        md_content += "### Strict Prompt\n"
        md_content += f"{res['strict_answer']}\n\n"
        md_content += f"Citations: {', '.join(res['strict_citations'])}\n\n"
        md_content += "### Permissive Prompt\n"
        md_content += f"{res['permissive_answer']}\n\n"
        md_content += "---\n\n"
        
    with open("./outputs/prompt_diff.md", 'w', encoding='utf-8') as f:
        f.write(md_content)
        
    print("Prompt comparison complete. Results saved to ./outputs/prompt_diff.md")
