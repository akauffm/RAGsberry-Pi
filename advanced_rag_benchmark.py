# Allows you to choose whether to use a flat FAISS index or an IVF index

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
TEXT_FILE_PATH = "my_document.txt" # Assumes this is a large file now
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# --- Index-specific Configuration ---
# For IVF Index
NLIST = 64          # Number of clusters/cells. For a few thousand vectors, 16-64 is a good start.
NPROBE = 8          # Number of nearby clusters to search at query time. Higher is more accurate but slower.

TOP_K = 3
OLLAMA_MODEL_NAME = "gemma3:1b"
OLLAMA_API_URL = "http://localhost:11434/api/generate"

# --- Chunking Configuration (can be tuned) ---
CHUNK_SIZE_CHARS = 1000
CHUNK_OVERLAP_CHARS = 100

# --- Helper functions ---

def preprocess_text(text: str) -> str:
    cleaned_text = re.sub(r' @(.)@ ', r'\1', text)
    return cleaned_text

def recursive_character_splitter(text: str, chunk_size: int, chunk_overlap: int):
    # This robust function remains the same
    if len(text) <= chunk_size: return [text]
    separators = ["\n\n", ". ", "\n", " ", ""]
    best_separator = next((s for s in separators if s in text), "")
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
        if current_chunk: current_chunk += best_separator + part
        else: current_chunk = part
    if current_chunk: final_chunks.append(current_chunk)
    return [c.strip() for c in final_chunks if c.strip()]


def create_and_save_index(text_file, index_path, chunks_path, model, index_type='flat'):
    """
    Creates and saves a FAISS index based on the specified type ('flat' or 'ivf').
    """
    print(f"--- Creating new FAISS index of type: {index_type.upper()} ---")
    start_time = time.time()

    try:
        with open(text_file, 'r', encoding='utf-8') as f:
            raw_text_content = f.read()
    except FileNotFoundError:
        print(f"Error: The source document '{text_file}' was not found.")
        return None, None

    print("Preprocessing text...")
    clean_text_content = preprocess_text(raw_text_content)
    
    chunks = recursive_character_splitter(clean_text_content, CHUNK_SIZE_CHARS, CHUNK_OVERLAP_CHARS)
    print(f"Document split into {len(chunks)} chunks.")
    if not chunks: return None, None

    print("Generating embeddings...")
    embeddings = model.encode(chunks, show_progress_bar=True).astype('float32')
    embedding_dim = model.get_sentence_embedding_dimension()

    # --- Index creation logic based on type ---
    if index_type == 'ivf':
        print(f"Building IVF index with nlist={NLIST}...")
        quantizer = faiss.IndexFlatL2(embedding_dim)
        index = faiss.IndexIVFFlat(quantizer, embedding_dim, NLIST)
        
        # Training the index on the data
        print("Training index...")
        if embeddings.shape[0] < NLIST:
            print(f"Warning: Number of vectors ({embeddings.shape[0]}) is less than nlist ({NLIST}). This is not ideal.")
        index.train(embeddings)
        print("Training complete.")

    elif index_type == 'flat':
        print("Building FlatL2 index...")
        index = faiss.IndexFlatL2(embedding_dim)
    
    else:
        raise ValueError(f"Unknown index_type: {index_type}")

    print("Adding vectors to the index...")
    index.add(embeddings)
    print(f"Index created. Total vectors: {index.ntotal}")

    faiss.write_index(index, index_path)
    with open(chunks_path, 'w', encoding='utf-8') as f:
        json.dump(chunks, f)
    
    end_time = time.time()
    print(f"Index and chunks saved successfully. Total creation time: {end_time - start_time:.2f} seconds.")
    
    return index, chunks
    

def load_existing_index(index_path, chunks_path):
    # This function works for any index type
    print("--- Loading existing FAISS index ---")
    try:
        index = faiss.read_index(index_path)
        with open(chunks_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        print(f"Index loaded successfully from '{index_path}'. Contains {index.ntotal} vectors.")
        return index, chunks
    except Exception as e:
        print(f"Error loading index files: {e}. Will attempt to create a new index.")
        return None, None


def main():
    parser = argparse.ArgumentParser(description="Advanced RAG benchmark for comparing FAISS index types.")
    parser.add_argument(
        '--index_type',
        type=str,
        choices=['flat', 'ivf'],
        default='flat',
        help="Type of FAISS index to use ('flat' for IndexFlatL2, 'ivf' for IndexIVFFlat)."
    )
    parser.add_argument('-v', '--verbose', action='store_true', help="Print the context sent to the LLM.")
    args = parser.parse_args()

    print(f"--- RAG Benchmark using {args.index_type.upper()} index ---")

    # --- Dynamic filenames based on index type ---
    FAISS_INDEX_PATH = f"my_document_{args.index_type}.faiss"
    CHUNKS_PATH = f"my_document_{args.index_type}_chunks.json"

    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print("Embedding model loaded.")

    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(CHUNKS_PATH):
        index, chunks = load_existing_index(FAISS_INDEX_PATH, CHUNKS_PATH)
    else:
        index, chunks = create_and_save_index(TEXT_FILE_PATH, FAISS_INDEX_PATH, CHUNKS_PATH, model, args.index_type)

    if index is None or chunks is None:
        print("Failed to load or create an index. Exiting.")
        return
        
    # --- IMPORTANT: Set search-time parameters for IVF index ---
    if args.index_type == 'ivf':
        index.nprobe = NPROBE
        print(f"IVF index search parameter set: nprobe = {NPROBE}")

    # --- Interactive Query Loop ---
    print("\n--- Ready to Chat! ---")
    
    while True:
        query = input("\nQuery (or 'quit'): ")
        if query.lower() in ['quit', 'exit']: break

        # STAGE 1: SEARCH & RETRIEVAL
        start_retrieval_time = time.time()
        query_embedding = model.encode([query]).astype('float32')
        D, I = index.search(query_embedding, TOP_K)
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
        #prompt = f"Context:\n{context_str}\n\nQuestion:\n{query}\n\nAnswer:"
        start_llm_time = time.time()
        try:
            response = requests.post(OLLAMA_API_URL, json={"model": OLLAMA_MODEL_NAME, "prompt": prompt, "stream": False}, timeout=120)
            response.raise_for_status()
            generated_text = response.json().get('response', '').strip()
        except requests.exceptions.RequestException as e:
            generated_text = f"Error communicating with Ollama: {e}"
        end_llm_time = time.time()
        llm_duration = end_llm_time - start_llm_time

        if args.verbose:
            print("\n\n--- [VERBOSE] Context Sent to LLM ---\n" + context_str + "\n---------------------------------------")

        print("\n--- Answer ---\n" + generated_text)
        print("\n--- Benchmarks ---")
        print(f"  Index Type:         {args.index_type.upper()}")
        print(f"  Search & Retrieval: {retrieval_duration:.4f} seconds")
        print(f"  LLM Generation:     {llm_duration:.4f} seconds")
        print(f"  Total Time:         {retrieval_duration + llm_duration:.4f} seconds")
        print("--------------------")

if __name__ == "__main__":
    main()
