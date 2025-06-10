import os
import time
import numpy as np
import faiss
import requests
import json
import re
import argparse
from sentence_transformers import SentenceTransformer

# --- Configuration ---
TEXT_FILE_PATH = "my_document.txt"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
FAISS_INDEX_PATH = "my_document_recursive.faiss"
CHUNKS_PATH = "my_document_recursive_chunks.json"

# --- Chunking Configuration ---
CHUNK_SIZE_CHARS = 1000
CHUNK_OVERLAP_CHARS = 100

TOP_K = 6
OLLAMA_MODEL_NAME = "gemma3:1b"
OLLAMA_API_URL = "http://localhost:11434/api/generate"


# --- NEW: Pre-processing Function ---
def preprocess_text(text: str) -> str:
    """
    Cleans the text by replacing custom-delimited punctuation
    with the standard punctuation mark.
    e.g., "1 @,@ 000" becomes "1,000"
    e.g., "3 @.@ 14" becomes "3.14"
    """
    # This regex finds a space, '@', any single character (the punctuation),
    # another '@', and a final space. It replaces this whole pattern
    # with just the captured character.
    # The r'\1' in the replacement refers to the first captured group, which is (.)
    cleaned_text = re.sub(r' @(.)@ ', r'\1', text)
    return cleaned_text


# --- Helper Functions (Recursive splitter is unchanged) ---
def recursive_character_splitter(text: str, chunk_size: int, chunk_overlap: int):
    """
    A more robust recursive text splitter. It first splits the text into large
    pieces, then recursively breaks down any pieces that are too large, and finally
    merges the pieces back together into chunks of the desired size.
    """
    if len(text) <= chunk_size:
        return [text]

    separators = ["\n\n", ". ", "\n", " ", ""]
    best_separator = ""
    for sep in separators:
        if sep in text:
            best_separator = sep
            break

    initial_splits = text.split(best_separator)
    processed_splits = []
    for part in initial_splits:
        if len(part) > chunk_size:
            processed_splits.extend(recursive_character_splitter(part, chunk_size, chunk_overlap))
        else:
            processed_splits.append(part)

    final_chunks = []
    current_chunk = ""
    for part in processed_splits:
        if len(current_chunk) + len(part) + len(best_separator) > chunk_size and current_chunk:
            final_chunks.append(current_chunk)
            overlap_start_index = max(0, len(current_chunk) - chunk_overlap)
            current_chunk = current_chunk[overlap_start_index:]
        
        if current_chunk:
            current_chunk += best_separator + part
        else:
            current_chunk = part

    if current_chunk:
        final_chunks.append(current_chunk)

    return [c.strip() for c in final_chunks if c.strip()]


def create_and_save_index(text_file, index_path, chunks_path, model):
    """Reads, PREPROCESSES, and chunks a text file to create and save a FAISS index."""
    print("--- Creating new FAISS index using Recursive Character Splitting ---")
    start_time = time.time()

    try:
        with open(text_file, 'r', encoding='utf-8') as f:
            raw_text_content = f.read()
    except FileNotFoundError:
        print(f"Error: The source document '{text_file}' was not found.")
        return None, None

    # --- Call the new preprocessing function ---
    print("Preprocessing text to handle custom punctuation...")
    clean_text_content = preprocess_text(raw_text_content)
    # -------------------------------------------

    # 1. Chunk the CLEANED text
    chunks = recursive_character_splitter(clean_text_content, CHUNK_SIZE_CHARS, CHUNK_OVERLAP_CHARS)
    print(f"Document split into {len(chunks)} chunks of ~{CHUNK_SIZE_CHARS} chars.")
    
    if not chunks:
        print("Warning: Text splitting resulted in zero chunks. Check document content and chunk size.")
        return None, None

    # 2. Generate embeddings
    print("Generating embeddings...")
    embeddings = model.encode(chunks, show_progress_bar=True)
    embedding_dim = model.get_sentence_embedding_dimension()

    # 3. Create and populate FAISS index
    print("Creating FAISS index...")
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(np.array(embeddings).astype('float32'))

    # 4. Save the index and chunks
    faiss.write_index(index, index_path)
    with open(chunks_path, 'w', encoding='utf-8') as f:
        json.dump(chunks, f)
    
    end_time = time.time()
    print(f"Index and chunks saved successfully to '{index_path}' and '{chunks_path}'.")
    print(f"BENCHMARK: Index creation took {end_time - start_time:.2f} seconds.")
    
    return index, chunks


def load_existing_index(index_path, chunks_path):
    """Loads a pre-existing FAISS index and its corresponding chunks."""
    print("--- Loading existing FAISS index ---")
    try:
        index = faiss.read_index(index_path)
        with open(chunks_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        print("Index and chunks loaded successfully.")
        return index, chunks
    except Exception as e:
        print(f"Error loading index files: {e}")
        print("Will attempt to create a new index.")
        return None, None


# --- Main Application (unchanged) ---
def main():
    parser = argparse.ArgumentParser(
        description="Interactive RAG benchmark with verbose output option."
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',  # This makes it a flag: if present, args.verbose is True
        help="Print the full context being sent to the LLM for each query."
    )
    args = parser.parse_args()

    print("--- Interactive RAG Benchmark with Recursive Splitting & Preprocessing ---")
    print(f"Using chunk size: {CHUNK_SIZE_CHARS} chars, overlap: {CHUNK_OVERLAP_CHARS} chars.")

    # Load the embedding model
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print("Embedding model loaded.")

    # Check if index exists, otherwise create it
    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(CHUNKS_PATH):
        index, chunks = load_existing_index(FAISS_INDEX_PATH, CHUNKS_PATH)
    else:
        index, chunks = create_and_save_index(TEXT_FILE_PATH, FAISS_INDEX_PATH, CHUNKS_PATH, model)

    if index is None or chunks is None:
        print("Failed to load or create an index. Exiting.")
        return

    # --- Interactive Query Loop ---
    print("\n--- Ready to Chat! ---")
    print("Enter your query below. Type 'quit' or 'exit' to stop.")
    
    while True:
        query = input("\nQuery: ")
        if query.lower() in ['quit', 'exit']:
            print("Exiting...")
            break

        # STAGE 1: SEARCH & RETRIEVAL
        start_retrieval_time = time.time()
        query_embedding = model.encode([query])
        D, I = index.search(np.array(query_embedding).astype('float32'), TOP_K)
        retrieved_chunks = [chunks[i] for i in I[0]]
        end_retrieval_time = time.time()
        retrieval_duration = end_retrieval_time - start_retrieval_time
        
        context_str = "\n\n".join(retrieved_chunks)

        # STAGE 2: LLM RESPONSE GENERATION
        prompt = f"""
Based on the following context, please answer the user's question.
If the context does not contain the answer, state that the information is not available.

Context:
{context_str}

Question:
{query}

Answer:
"""
        start_llm_time = time.time()
        try:
            response = requests.post(
                OLLAMA_API_URL,
                json={"model": OLLAMA_MODEL_NAME, "prompt": prompt, "stream": False},
                timeout=120
            )
            response.raise_for_status()
            generated_text = response.json().get('response', '').strip()
        except requests.exceptions.RequestException as e:
            generated_text = f"Error communicating with Ollama: {e}"

        end_llm_time = time.time()
        llm_duration = end_llm_time - start_llm_time

        if args.verbose:
            print("\n\n--- [VERBOSE] Context Sent to LLM ---")
            print(context_str)
            print("---------------------------------------")

        # --- Display Results ---
        print("\n--- Answer ---")
        print(generated_text)
        print("\n--- Benchmarks ---")
        print(f"  Search & Retrieval: {retrieval_duration:.4f} seconds")
        print(f"  LLM Generation:     {llm_duration:.4f} seconds")
        print(f"  Total Time:         {retrieval_duration + llm_duration:.4f} seconds")
        print("--------------------")

if __name__ == "__main__":
    main()