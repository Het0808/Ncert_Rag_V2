import os
import json
import logging
import numpy as np
from typing import List, Dict
from langchain_groq import ChatGroq

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HybridRetriever:
    def __init__(self, vectorstore, bm25_index, chunks, k=5):
        self.vectorstore = vectorstore
        self.bm25_index = bm25_index
        self.chunks = chunks
        self.k = k
        self.chunks_by_id = {c['chunk_id']: c for c in chunks}

    def retrieve(self, query: str, k: int = None) -> list[dict]:
        """Performs hybrid search (BM25 + Dense) with RRF fusion."""
        if k is None:
            k = self.k
            
        # BM25 step
        tokenized_query = query.lower().split()
        scores = self.bm25_index.get_scores(tokenized_query)
        matching_indices = [i for i in np.argsort(scores)[::-1] if scores[i] > 0]
        top_bm25_indices = matching_indices[:k*4]
        bm25_results = [(self.chunks[i]['chunk_id'], float(scores[i])) for i in top_bm25_indices]
        
        # Dense step
        # results = self.vectorstore.similarity_search_with_score(query, k=k*4) # VERIFY METHOD NAME
        results = self.vectorstore.similarity_search_with_score(query, k=k*4)
        dense_results = [(doc.metadata['chunk_id'], float(score)) for doc, score in results]
        
        # RRF fusion
        rrf_scores = {}
        
        # BM25 RRF
        for rank, (cid, _) in enumerate(bm25_results, 1):
            rrf_scores[cid] = rrf_scores.get(cid, 0) + 1 / (60 + rank)
            
        # Dense RRF
        for rank, (cid, _) in enumerate(dense_results, 1):
            rrf_scores[cid] = rrf_scores.get(cid, 0) + 1 / (60 + rank)
            
        # Sort by RRF score descending
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        top_ids = sorted_ids[:k]
        
        logger.info(f"Query: '{query}' | Hybrid results: {len(top_ids)}")
        return [self.chunks_by_id[cid] for cid in top_ids]

class RerankPipeline:
    def __init__(self, use_cohere: bool = True):
        self.use_cohere = use_cohere
        if self.use_cohere:
            try:
                import cohere
                api_key = os.getenv('COHERE_API_KEY')
                if not api_key:
                    raise ValueError("COHERE_API_KEY not found")
                self.client = cohere.Client(api_key)
            except Exception as e:
                logger.warning(f"Cohere init failed: {e}. Switching to local CrossEncoder.")
                self.use_cohere = False
                
        if not self.use_cohere:
            from sentence_transformers import CrossEncoder
            self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    def rerank(self, query: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
        """Reranks candidate chunks using Cohere or local CrossEncoder."""
        if not chunks:
            return []
            
        try:
            if self.use_cohere:
                # model='rerank-english-v3.0'
                results = self.client.rerank(
                    model='rerank-english-v3.0',
                    query=query,
                    documents=[c['content'] for c in chunks],
                    top_n=top_k
                )
                reranked_chunks = []
                # Map result indices back to original chunks
                for result in results.results:
                    idx = result.index
                    chunk = chunks[idx].copy()
                    chunk['rerank_score'] = result.relevance_score
                    reranked_chunks.append(chunk)
                return reranked_chunks
            else:
                pairs = [[query, c['content']] for c in chunks]
                scores = self.cross_encoder.predict(pairs)
                # Attach scores
                for i, score in enumerate(scores):
                    chunks[i]['rerank_score'] = float(score)
                # Sort by score descending
                sorted_chunks = sorted(chunks, key=lambda x: x['rerank_score'], reverse=True)
                return sorted_chunks[:top_k]
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return chunks[:top_k]

class EnhancedRetriever:
    def __init__(self, hybrid_retriever, reranker, use_multiquery: bool = True):
        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker
        self.use_multiquery = use_multiquery
        if self.use_multiquery:
            api_key = os.getenv("GROQ_API_KEY")
            if api_key and api_key.strip() and not api_key.startswith("your_"):
                self.llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
            else:
                logger.warning("GROQ_API_KEY not set or invalid. Disabling multi-query expansion.")
                self.use_multiquery = False

    def retrieve(self, query: str, k: int = 5) -> list[dict]:
        """Main retrieval entry point with optional multiquery and reranking."""
        if self.use_multiquery:
            try:
                prompt = (
                    f"Generate exactly 3 alternative phrasings of this question. "
                    f"Return only the 3 questions, one per line, no numbering:\n{query}"
                )
                response = self.llm.invoke(prompt)
                variants = [q.strip() for q in response.content.split('\n') if q.strip()]
                # [query] + variants[:2] (total 3 including original)
                queries = [query] + variants[:2]
            except Exception as e:
                logger.error(f"Multi-query generation failed: {e}")
                queries = [query]
        else:
            queries = [query]
            
        all_candidates = []
        seen_ids = set()
        
        for q in queries:
            candidates = self.hybrid_retriever.retrieve(q, k=20)
            for c in candidates:
                if c['chunk_id'] not in seen_ids:
                    all_candidates.append(c)
                    seen_ids.add(c['chunk_id'])
                    
        return self.reranker.rerank(query, all_candidates, top_k=k)

def run_retrieval_test(retriever, test_queries: list[str]) -> list[dict]:
    """Runs a series of tests and logs results."""
    results = []
    output_dir = pathlib.Path("./outputs")
    output_dir.mkdir(exist_ok=True)
    
    for query in test_queries:
        top_chunks = retriever.retrieve(query, k=5)
        record = {
            "query": query,
            "top_chunk_id": top_chunks[0]['chunk_id'] if top_chunks else None,
            "top_content_preview": top_chunks[0]['content'][:200] if top_chunks else "",
            "all_chunk_ids": [c['chunk_id'] for c in top_chunks]
        }
        results.append(record)
        
    with open('./outputs/retrieval_log.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
        
    return results
