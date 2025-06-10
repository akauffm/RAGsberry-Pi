import os
import time
import numpy as np
import faiss
import requests
import json
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# --- Configuration ---
# Stage 1: Indexing Configuration
TEXT_FILE_PATH = "my_document.txt"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
FAISS_INDEX_PATH = "my_document.faiss"

# Stage 2: Search & Retrieval Configuration
TOP_K = 3 # Number of relevant chunks to retrieve

# Stage 3: LLM Response Configuration
OLLAMA_MODEL_NAME = "gemma3:1b" # The model you pulled with "ollama pull"
OLLAMA_API_URL = "http://localhost:11434/api/generate"

# --- Helper Functions ---

def simple_text_splitter(text, chunk_size=3, chunk_overlap=1):
    """
    A very simple text splitter that splits by sentences.
    A more robust solution would use LangChain's RecursiveCharacterTextSplitter.
    """
    sentences = text.replace("\n", " ").split('. ')
    sentences = [s.strip() for s in sentences if s.strip()]
    
    chunks = []
    for i in range(0, len(sentences), chunk_size - chunk_overlap):
        chunk = ". ".join(sentences[i:i + chunk_size])
        if chunk:
            chunks.append(chunk + ".")
            
    return chunks

# --- Main Benchmarking Script ---

def main():
    print("--- RAG Performance Benchmark on Raspberry Pi ---")

    # ==================================================================
    # STAGE 1: INDEXING
    # ==================================================================
    print("\n--- STAGE 1: INDEXING ---")
    
    # Load the embedding model
    # The first time this runs, it will download the model. This is a one-time cost.
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    embedding_dim = model.get_sentence_embedding_dimension()
    print(f"Model loaded. Embedding dimension: {embedding_dim}")

    start_time_indexing = time.time()

    # Load and chunk the document
    try:
        with open(TEXT_FILE_PATH, 'r', encoding='utf-8') as f:
            text_content = f.read()
    except FileNotFoundError:
        print(f"Error: The file '{TEXT_FILE_PATH}' was not found.")
        return

    # Using a simple sentence-based chunking strategy
    chunks = simple_text_splitter(text_content)
    print(f"Document split into {len(chunks)} chunks.")

    # Generate embeddings for each chunk
    print("Generating embeddings for all chunks...")
    chunk_embeddings = model.encode(chunks, show_progress_bar=True)
    
    # Create a FAISS index
    print("Creating FAISS index...")
    # Using IndexFlatL2 - a simple L2 distance (Euclidean) index
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(np.array(chunk_embeddings).astype('float32'))

    # Save the index and the chunks
    # We need to save the chunks themselves to retrieve the text later
    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(FAISS_INDEX_PATH + ".json", 'w') as f:
        json.dump(chunks, f)

    end_time_indexing = time.time()
    indexing_duration = end_time_indexing - start_time_indexing
    
    print(f"FAISS index created and saved to '{FAISS_INDEX_PATH}'")
    print("-----------------------------------------------------")
    print(f"BENCHMARK: Indexing took {indexing_duration:.4f} seconds.")
    print("-----------------------------------------------------")


    # ==================================================================
    # STAGE 2: SEARCH & RETRIEVAL
    # ==================================================================
    print("\n--- STAGE 2: SEARCH & RETRIEVAL ---")
    
    query = "What was the Sinclair Sovereign and how much did it cost?"
    print(f"Sample Query: '{query}'")

    start_time_retrieval = time.time()

    # Embed the query
    query_embedding = model.encode([query])

    # Search the FAISS index
    # D: distances, I: indices of the nearest neighbors
    D, I = index.search(np.array(query_embedding).astype('float32'), TOP_K)

    # Retrieve the actual text chunks
    retrieved_chunks = [chunks[i] for i in I[0]]
    
    end_time_retrieval = time.time()
    retrieval_duration = end_time_retrieval - start_time_retrieval

    print(f"\nTop {TOP_K} relevant chunks found:")
    for i, chunk in enumerate(retrieved_chunks):
        print(f"  {i+1}. {chunk}")

    print("-----------------------------------------------------")
    print(f"BENCHMARK: Search & Retrieval took {retrieval_duration:.4f} seconds.")
    print("-----------------------------------------------------")


    # ==================================================================
    # STAGE 3: LLM RESPONSE GENERATION
    # ==================================================================
    print("\n--- STAGE 3: LLM RESPONSE GENERATION ---")
    
    # Prepare the context for the LLM
    context_str = "\n\n".join(retrieved_chunks)

    # Create the prompt
    prompt = f"""
Based on the following context, please answer the user's question.
If the context does not contain the answer, state that the information is not available in the provided context.

Context:
{context_str}

Question:
{query}

Answer:
"""

    print("Sending request to Ollama...")
    start_time_llm = time.time()

    try:
        response = requests.post(
            OLLAMA_API_URL,
            json={
                "model": OLLAMA_MODEL_NAME,
                "prompt": prompt,
                "stream": False # We want the full response at once
            },
            timeout=120 # Add a timeout
        )
        response.raise_for_status() # Raise an exception for bad status codes
        
        generated_text = response.json().get('response', '').strip()

    except requests.exceptions.RequestException as e:
        print(f"\nError connecting to Ollama: {e}")
        print("Please make sure Ollama is running and the model is available.")
        generated_text = "Error: Could not get a response from the LLM."

    end_time_llm = time.time()
    llm_duration = end_time_llm - start_time_llm

    print("\nLLM Generated Answer:")
    print(generated_text)

    print("-----------------------------------------------------")
    print(f"BENCHMARK: LLM Generation took {llm_duration:.4f} seconds.")
    print("-----------------------------------------------------")
    
    print("\n--- Benchmark Summary ---")
    print(f"  Indexing:            {indexing_duration:.4f} seconds")
    print(f"  Search & Retrieval:  {retrieval_duration:.4f} seconds")
    print(f"  LLM Generation:      {llm_duration:.4f} seconds")
    print("--------------------------")
    print(f"  Total RAG Pipeline:  {retrieval_duration + llm_duration:.4f} seconds (excluding one-time indexing)")


if __name__ == "__main__":
    main()
