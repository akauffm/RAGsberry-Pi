# Running a RAG on a Raspberry Pi 5

Local RAG running on a Raspberry Pi 5 with 8GB of RAM,\
LLM is Gemma3:1b also running locally on the Pi \
(**make sure the sound is on**)

https://github.com/user-attachments/assets/c78c94ca-4f79-44f1-add1-f3b7388345df

[ktomanek](https://github.com/ktomanek/) and I have been working on understanding the limitations of running largish models on smallish hardware. I didn't know a whole lot about the inner workings of RAGs and figured that getting one running on a Raspberry Pi would provide an excellent introduction.

My goal was to build an offline version of Wikipedia that I could talk to using my choice of LLM served locally via [Ollama](https://ollama.com) and the [modular voice agent](https://github.com/akauffm/edge_voice_agent) we built previously. To avoid having to index a 50GB text file I used wikitext 2, a representative [~13MB chunk of Wikipedia](https://www.kaggle.com/datasets/vanshitg/textdata?select=wiki.train.txt). Though the index architecture doesn't matter for such a small dataset, the full 50GB text of Wikipedia would result in approximately 36M vectors with the chunking settings I used. Search would be significantly slower using a flat index but still around 100ms using IVF, so I implemented both architectures (switch between them with a flag).

I initially tried using a [framework](https://github.com/open-webui/open-webui) that included RAG capabilities but when things didn't work as expected, debugging proved much harder than implementing the RAG from scratch, which forced me to learn about vector indexes and a bunch of other things.

## TL;DR
I learned:
1. There are lots of variables to play with when it comes to RAG—vector index architecture, chunk size and overlap in the vector index, the sentence embedding model, the chunking method for the input text (these affected the model's ability to return relevant information, but not its overall performance), number of results returned (this made the biggest difference on performance), LLM used (unsurprisingly, larger models hallucinated less).
2. [FAISS](https://github.com/facebookresearch/faiss) is the way to go for searching a vector database on the Pi. There is technically a faster in-memory implementation of the same algorithm, but the search takes around 100ms, so not worth further optimization.
3. As noted above, the single biggest factor is the length of the query passed to the LLM. The response time scales linearly with context length, so passing in more than a single search result extended the response time to between 10 and 40 seconds, depending on the model.
4. **Most interesting:** Some response caching happens behind the scenes, so if you ask the same thing quickly twice in a row, the second response is much faster than the first. But we found no straightforward way to configure or interact in any meaningful way with that cache. Seems like an area for further investigation.

## Files
- [**Use this one**] *interactive_rag_benchmark.py*: Incorporates improvements from the other versions
- *rag_benchmark.py*: basic version with single, fixed prompt
- *recursive_rag_benchmark.py*: basic version with added text cleaning and recursive chunking (rather than more naive sentence or token chunking)—interesting experiment, no noticeable performance differences
- *advanced_rag_benchmark.py*: basic version plus choice between flat and IVF indexes
