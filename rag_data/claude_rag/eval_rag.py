import random
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
#from chat_rag import retrieve, generate_answer
from chat_rag import retrieve, build_context, build_prompt, query_ollama, embed_query

# ================= CONFIG =================
INDEX_PATH = "fish.index"
RECORDS_PATH = "records.pkl"
EMBED_MODEL = "all-MiniLM-L6-v2"
TOP_K = 5                     # <-- Recall@5
SAMPLE_SIZE = 10
SIM_THRESHOLD = 0.60
# ==========================================


def load_index():
    return faiss.read_index(INDEX_PATH)


def load_records():
    with open(RECORDS_PATH, "rb") as f:
        return pickle.load(f)


'''# ---------------- RETRIEVAL ----------------
def retrieve(query, model, index, records):
    query_vec = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    scores, indices = index.search(query_vec, TOP_K)

    retrieved_docs = []
    for score, idx in zip(scores[0], indices[0]):
        if idx != -1 and score >= SIM_THRESHOLD:
            retrieved_docs.append(records[idx])

    return retrieved_docs, scores[0], indices[0]


# ---------------- GENERATION ----------------
# Replace this with your RAG LLM call
def generate_answer(query, contexts):
    # Simple baseline: concatenate context
    context_text = " ".join([c["Description"] for c in contexts])
    return context_text[:500]  # simulate generation

'''
# ------------- GENERATION METRICS -------------

def compute_faithfulness(answer, contexts, model):
    """
    Measures how grounded answer is in retrieved context
    """
    context_text = " ".join([c["Description"] for c in contexts])

    emb_answer = model.encode([answer], normalize_embeddings=True)
    emb_context = model.encode([context_text], normalize_embeddings=True)

    score = cosine_similarity(emb_answer, emb_context)[0][0]
    return score


def compute_relevance(query, answer, model):
    """
    Measures semantic similarity between query and answer
    """
    emb_query = model.encode([query], normalize_embeddings=True)
    emb_answer = model.encode([answer], normalize_embeddings=True)

    score = cosine_similarity(emb_query, emb_answer)[0][0]
    return score


def compute_hallucination_rate(faithfulness_score, threshold=0.5):
    """
    If faithfulness is low → hallucination likely
    """
    return 1 if faithfulness_score < threshold else 0


# ---------------- EVALUATION ----------------
def evaluate():
    index = load_index()
    records = load_records()
    model = SentenceTransformer(EMBED_MODEL)

    sample = random.sample(records, SAMPLE_SIZE)

    recall_hits = 0
    reciprocal_ranks = []

    faithfulness_scores = []
    relevance_scores = []
    hallucination_flags = []

    '''for item in sample:
        query = item["Species Name"]
        true_name = item["Species Name"]

        retrieved_docs, scores, indices = retrieve(query, model, index, records)

        # ---------- RETRIEVAL METRICS ----------
        found = False

        for rank, idx in enumerate(indices, start=1):
            if idx != -1 and records[idx]["Species Name"] == true_name:
                recall_hits += 1
                reciprocal_ranks.append(1 / rank)
                found = True
                break

        if not found:
            reciprocal_ranks.append(0)

        # ---------- GENERATION ----------
        if len(retrieved_docs) > 0:
            answer = generate_answer(query, retrieved_docs)

            faith = compute_faithfulness(answer, retrieved_docs, model)
            rel = compute_relevance(query, answer, model)
            hall = compute_hallucination_rate(faith)

            faithfulness_scores.append(faith)
            relevance_scores.append(rel)
            hallucination_flags.append(hall)'''
    
    for item in sample:
        query = item["Species Name"]
        true_name = item["Species Name"]

        # -------- RETRIEVAL (REAL) --------
        #retrieved = retrieve(query, model, index, records, top_k=5)
        query_vec = embed_query(query, model)
        retrieved = retrieve(query_vec, index, records, top_k=5)

        found = False
        for rank, r in enumerate(retrieved, start=1):
            if r["Species Name"] == true_name:
                recall_hits += 1
                reciprocal_ranks.append(1 / rank)
                found = True
                break

        if not found:
            reciprocal_ranks.append(0)

        # -------- GENERATION (REAL OLLAMA CALL) --------
        #answer = generate_answer(query, retrieved)
        context = build_context(retrieved)
        prompt = build_prompt(context, query)
        answer = query_ollama(prompt)

        faith = compute_faithfulness(answer, retrieved, model)
        rel = compute_relevance(query, answer, model)
        hall = compute_hallucination_rate(faith)

        faithfulness_scores.append(faith)
        relevance_scores.append(rel)
        hallucination_flags.append(hall)

    # ---------------- FINAL METRICS ----------------
    total = len(sample)

    recall_at_5 = recall_hits / total
    mrr = sum(reciprocal_ranks) / total

    avg_faithfulness = np.mean(faithfulness_scores)
    avg_relevance = np.mean(relevance_scores)
    hallucination_rate = np.mean(hallucination_flags)

    print("\n=========== RAG EVALUATION ===========")
    print(f"Total Queries        : {total}")
    print(f"Recall@5             : {recall_at_5:.4f}")
    print(f"MRR                  : {mrr:.4f}")
    print("--------------- Generation ---------------")
    print(f"Faithfulness         : {avg_faithfulness:.4f}")
    print(f"Relevance Score      : {avg_relevance:.4f}")
    print(f"Hallucination Rate   : {hallucination_rate:.4f}")
    print("==========================================")


if __name__ == "__main__":
    evaluate()