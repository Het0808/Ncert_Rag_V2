import os
import re
import json
import logging
import pathlib
from dotenv import load_dotenv
import fitz
from langchain.docstore.document import Document

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

def load_pdf(pdf_path: str, output_format: str) -> list:
    """
    Loads a PDF using fitz (PyMuPDF) and returns a list of LangChain Documents.
    """
    if not os.path.exists(pdf_path):
        logger.error(f"File not found: {pdf_path}")
        return []
        
    doc = fitz.open(pdf_path)
    documents = []
    
    for i, page in enumerate(doc):
        text = page.get_text("text")
        # Clean up text
        text = re.sub(r'\s+', ' ', text).strip()
        
        if text:
            metadata = {
                "source": pdf_path,
                "page": i + 1
            }
            documents.append(Document(page_content=text, metadata=metadata))
            
    logger.info(f"Loaded {len(documents)} pages from {pdf_path}")
    return documents

def load_corpus(pdf_dir: str, chapter_files: list, output_format: str) -> dict:
    """
    Loads multiple PDFs into a corpus dictionary.
    """
    corpus = {}
    total_docs = 0
    
    for filename in chapter_files:
        pdf_path = str(pathlib.Path(pdf_dir) / filename)
        docs = load_pdf(pdf_path, output_format)
        corpus[filename] = docs
        total_docs += len(docs)
        
    logger.info(f"Total documents loaded across all files: {total_docs}")
    return corpus

if __name__ == "__main__":
    pdf_directory = "./data/pdfs/"
    chapters = ["iesc107.pdf", "iesc108.pdf", "iesc110.pdf"]
    
    # Load all three chapters in "markdown" format
    corpus_dict = load_corpus(pdf_directory, chapters, "markdown")
    
    total_count = sum(len(docs) for docs in corpus_dict.values())
    print(f"Total document count: {total_count}")
