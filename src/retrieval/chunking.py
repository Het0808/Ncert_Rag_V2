import re
import json
import hashlib
import pathlib
import tiktoken
from src.ingestion.ingestion import load_corpus

class StructureAwareChunker:
    def __init__(self, max_tokens=250, overlap_tokens=30):
        self.encoding = tiktoken.get_encoding("cl100k_base")
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    def chunk_document(self, markdown_text: str, metadata: dict) -> list[dict]:
        """Splits markdown text into sections by headings and processes each."""
        # Split by headings: regex provided in requirements
        # TEST THIS REGEX BEFORE USING IN PRODUCTION
        pattern = r'(^|\n)(#+\s+.+)'
        sections = re.split(pattern, markdown_text)
        
        chunks = []
        current_heading = "Introduction"
        
        # re.split with 2 capturing groups will yield: 
        # [prefix, group1, group2, suffix, ...]
        # Here: [text_before, newline, heading, text_after, ...]
        
        i = 0
        while i < len(sections):
            text = sections[i].strip()
            if text:
                chunks.extend(self._process_section(text, current_heading, metadata))
            
            if i + 2 < len(sections):
                current_heading = sections[i + 2].strip()
                i += 3
            else:
                break
                
        return chunks

    def _process_section(self, text: str, heading: str, metadata: dict) -> list[dict]:
        """Classifies section and chunks accordingly."""
        full_text = f"{heading}\n{text}"
        token_count = len(self.encoding.encode(full_text))
        
        if "example" in text.lower():
            content_type = "worked_example"
            # Keep entire heading + text as one chunk regardless of limit as per v2 fix
            return [self._create_chunk(full_text, content_type, metadata)]
            
        elif text.count('|') > 6:
            content_type = "table"
            if token_count <= self.max_tokens * 1.5:
                return [self._create_chunk(full_text, content_type, metadata)]
            # If over 1.5x, requirements don't explicitly say what to do for tables, 
            # but usually they are kept together or fallback. 
            # I will keep together to be safe unless specified.
            return [self._create_chunk(full_text, content_type, metadata)]
            
        else:
            content_type = "prose"
            if token_count <= self.max_tokens:
                return [self._create_chunk(full_text, content_type, metadata)]
            
            # Split on sentence boundaries: regex provided in requirements
            # TEST THIS REGEX BEFORE USING IN PRODUCTION
            sentences = re.split(r'(?<=[.!?])\s+', text)
            
            chunks = []
            current_chunk_sentences = [heading]
            current_tokens = len(self.encoding.encode(heading))
            
            for sentence in sentences:
                sentence_tokens = len(self.encoding.encode(sentence))
                if current_tokens + sentence_tokens > self.max_tokens and current_chunk_sentences:
                    chunks.append(self._create_chunk(" ".join(current_chunk_sentences), content_type, metadata))
                    # Overlap logic (simple version: start next with last sentence if requested, 
                    # but requirements say 'start new chunk' after max_tokens)
                    current_chunk_sentences = [sentence]
                    current_tokens = sentence_tokens
                else:
                    current_chunk_sentences.append(sentence)
                    current_tokens += sentence_tokens
            
            if current_chunk_sentences:
                chunks.append(self._create_chunk(" ".join(current_chunk_sentences), content_type, metadata))
            
            return chunks

    def _create_chunk(self, text: str, content_type: str, metadata: dict) -> dict:
        """Generates a chunk dictionary with required fields and MD5-based ID."""
        token_count = len(self.encoding.encode(text))
        
        # Generate chunk_id
        hash_str = hashlib.md5(text.encode()).hexdigest()[:12]
        source_stem = pathlib.Path(metadata.get("source", "unknown")).stem
        chunk_id = f"{source_stem}_{hash_str}"
        
        return {
            "chunk_id": chunk_id,
            "content": text,
            "content_type": content_type,
            "metadata": {
                "source": metadata.get("source"),
                "page": metadata.get("page"),
                "token_count": token_count
            }
        }

def process_corpus(pdf_docs: dict, output_path: str) -> list[dict]:
    """Processes entire corpus and saves to JSON."""
    chunker = StructureAwareChunker()
    all_chunks = []
    
    counts = {"prose": 0, "worked_example": 0, "table": 0}
    
    for filename, documents in pdf_docs.items():
        for doc in documents:
            # OpenDataLoaderPDFLoader documents usually have .page_content and .metadata
            # Metadata usually contains 'source' and 'page'
            markdown_text = doc.page_content
            doc_metadata = doc.metadata
            
            chunks = chunker.chunk_document(markdown_text, doc_metadata)
            all_chunks.extend(chunks)
            
            for c in chunks:
                counts[c["content_type"]] = counts.get(c["content_type"], 0) + 1
                
    # Save to JSON
    output_dir = pathlib.Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, indent=2)
        
    print(f"Summary by content type:")
    for ctype, count in counts.items():
        print(f"  {ctype}: {count}")
        
    return all_chunks

if __name__ == "__main__":
    pdf_directory = "./data/pdfs/"
    chapters = ["iesc107.pdf", "iesc108.pdf", "iesc110.pdf"]
    output_file = "./data/processed/chunks.json"
    
    # Loads three chapters from ingestion
    corpus_dict = load_corpus(pdf_directory, chapters, "markdown")
    
    # Process corpus
    final_chunks = process_corpus(corpus_dict, output_file)
    
    # Counts
    p_count = sum(1 for c in final_chunks if c["content_type"] == "prose")
    we_count = sum(1 for c in final_chunks if c["content_type"] == "worked_example")
    t_count = sum(1 for c in final_chunks if c["content_type"] == "table")
    
    print(f"Created {len(final_chunks)} chunks: {p_count} prose, {we_count} worked_example, {t_count} table")
