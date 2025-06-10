import os
import time
import numpy as np
import faiss
import requests
import json
from sentence_transformers import SentenceTransformer

# --- Configuration ---
TEXT_FILE_PATH = "my_document.txt"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
FAISS_INDEX_PATH = "my_document.faiss"
CHUNKS_PATH = "my_document_chunks.json" # Separate file for the text chunks

TOP_K = 3 # Number of relevant chunks to retrieve
OLLAMA_MODEL_NAME = "gemma3:12b" # The model you pulled with "ollama pull"
OLLAMA_API_URL = "http://localhost:11434/api/generate"

# --- Helper Functions for Indexing and Loading ---

def simple_text_splitter(text, chunk_size=5, chunk_overlap=1):
    """A very simple text splitter that splits by sentences."""
    sentences = text.replace("\n", " ").split('. ')
    sentences = [s.strip() for s in sentences if s.strip()]
    
    chunks = []
    for i in range(0, len(sentences), chunk_size - chunk_overlap):
        chunk = ". ".join(sentences[i:i + chunk_size])
        if chunk:
            chunks.append(chunk + ".")
    return chunks

def create_and_save_index(text_file, index_path, chunks_path, model):
    """Reads a text file, creates embeddings, and saves the FAISS index and chunks."""
    print("--- Creating new FAISS index ---")
    start_time = time.time()

    try:
        with open(text_file, 'r', encoding='utf-8') as f:
            text_content = f.read()
    except FileNotFoundError:
        print(f"Error: The source document '{text_file}' was not found.")
        return None, None

    # 1. Chunk the text
    chunks = simple_text_splitter(text_content)
    print(f"Document split into {len(chunks)} chunks.")

    # 2. Generate embeddings
    print("Generating embeddings (this may take a moment)...")
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

# --- Main Application ---

def main():
    print("--- Interactive RAG Benchmark on Raspberry Pi ---")

    # Load the embedding model (needed for both indexing and querying)
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
    try:
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    except Exception as e:
        print(f"Could not load Sentence Transformer model. Error: {e}")
        print("Please check your internet connection or the model name.")
        return
    print("Embedding model loaded.")

    # Check if index exists, otherwise create it
    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(CHUNKS_PATH):
        index, chunks = load_existing_index(FAISS_INDEX_PATH, CHUNKS_PATH)
        if index is None: # If loading failed, fallback to creating
             index, chunks = create_and_save_index(TEXT_FILE_PATH, FAISS_INDEX_PATH, CHUNKS_PATH, model)
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

        # STAGE 1: SEARCH & RETRIEVAL (BENCHMARKED)
        start_retrieval_time = time.time()
        
        query_embedding = model.encode([query])
        D, I = index.search(np.array(query_embedding).astype('float32'), TOP_K)
        retrieved_chunks = [chunks[i] for i in I[0]]
        
        end_retrieval_time = time.time()
        retrieval_duration = end_retrieval_time - start_retrieval_time
        
        context_str = "\n\n".join(retrieved_chunks)

        # STAGE 2: LLM RESPONSE GENERATION (BENCHMARKED)
        prompt = f"""
Based on the following context, please answer the user's question.
If the context does not contain the answer, state that the information is not available in the provided context.

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
