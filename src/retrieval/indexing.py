import json
import logging
import chromadb
from rank_bm25 import BM25Okapi
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma # UPDATED FOR LANGCHAIN 0.3

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def build_chroma_index(chunks: list[dict], persist_dir: str, collection_name: str, force_rebuild: bool = False):
    """Builds or loads a Chroma vectorstore index."""
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path=persist_dir)
    
    if force_rebuild:
        try:
            client.delete_collection(collection_name)
            logger.info(f"Deleted existing collection '{collection_name}' for fresh rebuild.")
        except Exception:
            pass
            
    try:
        # Check if collection exists
        client.get_collection(collection_name)
        logger.info("Collection exists, skipping embedding")
        # Load existing
        vectorstore = Chroma(
            client=client,
            collection_name=collection_name,
            embedding_function=embeddings
        )
    except Exception:
        logger.info(f"Creating new collection: {collection_name}")
        texts = [c['content'] for c in chunks]
        metadatas = [
            {
                "chunk_id": c['chunk_id'],
                "content_type": c['content_type'],
                "source": c['metadata']['source'],
                "page": str(c['metadata']['page']) # Page as string
            } for c in chunks
        ]
        ids = [c['chunk_id'] for c in chunks]
        
        vectorstore = Chroma.from_texts(
            texts=texts,
            metadatas=metadatas,
            ids=ids,
            collection_name=collection_name,
            persist_directory=persist_dir,
            embedding=embeddings
        )
    
    # Log how many documents are in the collection
    count = vectorstore._collection.count()
    logger.info(f"Chroma collection '{collection_name}' contains {count} documents.")
    return vectorstore

def build_bm25_index(chunks: list[dict]):
    """Builds a BM25 index from chunks in memory."""
    # Tokenize: each chunk's content lowercased and split on whitespace
    tokenized_corpus = [c['content'].lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    logger.info(f"BM25 index built on {len(chunks)} documents")
    return bm25

def load_indexes(persist_dir: str, collection_name: str, chunks: list[dict]):
    """Loads existing Chroma vectorstore and rebuilds BM25."""
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path=persist_dir)
    
    # Loads existing Chroma vectorstore (does NOT re-embed)
    vectorstore = Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=embeddings
    )
    
    bm25 = build_bm25_index(chunks)
    return vectorstore, bm25

if __name__ == "__main__":
    chunks_path = "./data/processed/chunks.json"
    persist_directory = "./vectorstore/chroma_db"
    coll_name = "parishiksha_v2"
    
    with open(chunks_path, 'r', encoding='utf-8') as f:
        all_chunks = json.load(f)
        
    # Calls build_chroma_index()
    vs = build_chroma_index(all_chunks, persist_directory, coll_name, force_rebuild=True)
    
    # Calls build_bm25_index()
    bm = build_bm25_index(all_chunks)
    
    print("Indexing complete")
