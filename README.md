# RAG Question Answering over a PDF

A Retrieval-Augmented Generation (RAG) system that answers questions about a PDF using only what the document says, and replies "I don't know" when the answer isn't there.

**Results:** removing noisy table-of-contents pages raised retrieval hit rate from **38% to 50%**, and the system correctly refused **3/3** questions the document cannot answer.

## Problem

Large language models don't know your documents, and when they don't know something they often make up a confident answer (hallucination). RAG solves this by first retrieving the relevant passages from the document, then asking the LLM to answer only from them, with the source pages attached.

## Pipeline

1. **Load** the PDF page by page (`PyPDFLoader`).
2. **Clean** the text and remove table-of-contents pages, which list every topic and match almost any question.
3. **Chunk** pages into 1,000-character pieces with 200 characters of overlap (`RecursiveCharacterTextSplitter`).
4. **Embed** each chunk with `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional vectors).
5. **Store** the vectors in a **FAISS** index.
6. **Retrieve** the 3 chunks most similar to the question.
7. **Generate** an answer with **gpt-oss-120b** served through the **Groq API** (temperature 0), using a prompt that allows only the retrieved context and requires a fixed refusal sentence when the answer is missing.
8. **Return** the answer with its source pages.

Document used: the TutorialsPoint *Machine Learning Tutorial* (PDF).

## Evaluation

A small hand-written test set of 11 questions:

- **8 answerable questions**, each labelled with an evidence phrase from the passage that answers it. A sanity check confirms every phrase exists in the document.
- **3 unanswerable questions** that the document does not cover (for example, "What is the capital of France?").

### Retrieval

- **Hit Rate@k:** share of answerable questions where at least one of the top-k chunks contains the evidence.
- **MRR (Mean Reciprocal Rank):** rewards finding the evidence early (rank 1 scores 1, rank 2 scores 0.5, rank 3 scores 0.33).

| Setup | k | Hit rate | MRR |
|---|---|---|---|
| Baseline (raw pages) | 3 | 0.38 | 0.38 |
| Clean pages (table of contents removed) | 3 | 0.50 | 0.35 |
| Clean pages | 5 | 0.50 | 0.35 |

**Takeaways**

- Removing table-of-contents pages found the correct passage for one more question (3 → 4 of 8).
- Increasing k from 3 to 5 did not help, so the missed passages are not just below the cut-off. The limitation is the quality of the semantic matching, not the number of chunks retrieved.

### Grounding (refusals)

| Question | Answer | Refused |
|---|---|---|
| What is the capital of France? | I don't know based on the provided document. | Yes |
| What was TutorialsPoint's revenue in 2023? | I don't know based on the provided document. | Yes |
| How does the attention mechanism in transformers work? | I don't know based on the provided document. | Yes |

The model knows from its training that the capital of France is Paris, but the document doesn't say it, so refusing shows the answers come from the retrieved context rather than the model's memory.

## Limitations

- **Small test set** (8 answerable and 3 unanswerable questions), so the scores are indicative rather than statistically strong.
- **Strict evidence labels:** a chunk that answers in different words counts as a miss, so retrieval scores may be slightly pessimistic.
- **Answer quality is not scored automatically:** retrieval and refusals are measured, while answer correctness was checked by reading.
- **Simple retrieval:** a single document and plain similarity search.

## Next steps

- A stronger embedding model
- Hybrid search (keyword BM25 plus embeddings) and a re-ranking model
- A larger test set and automatic answer scoring (LLM-as-judge or RAGAS)

## How to run

1. Open `RAG.ipynb` in Google Colab or Kaggle.
2. Get a free API key from [console.groq.com](https://console.groq.com/keys) and add it as a secret named `GROQ_API_KEY`, with notebook access turned on.
3. Run all cells. The notebook picks an available Groq chat model automatically.

## Tools

Python, LangChain, FAISS, sentence-transformers, Groq API, pandas

## Author

Yehia El Salmy
[LinkedIn](https://linkedin.com/in/yehia-el-salmy-59b5ba307) | [GitHub](https://github.com/yehiaelsalmi)
