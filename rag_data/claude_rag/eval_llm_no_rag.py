import pandas as pd
import requests
import time
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "phi3"

# Load embedding model for similarity scoring
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ---------- Ask Phi3 ----------
def ask_llm(question):
    prompt = f"Answer clearly and factually:\n{question}"
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False
        }
    )
    return response.json()["response"]

# ---------- Similarity ----------
def semantic_similarity(a, b):
    emb1 = embedder.encode([a])
    emb2 = embedder.encode([b])
    return cosine_similarity(emb1, emb2)[0][0]

# ---------- Faithfulness ----------
# checks if answer stays close to ground truth
def faithfulness_score(answer, ground_truth):
    return semantic_similarity(answer, ground_truth)

# ---------- Relevance ----------
# checks if answer relates to question
def relevance_score(answer, question):
    return semantic_similarity(answer, question)

# ---------- Hallucination ----------
# If similarity to ground truth is low → hallucination high
def hallucination_rate(faithfulness):
    return 1 - faithfulness

# ---------- MAIN EVAL ----------
def run_evaluation(csv_file):
    df = pd.read_csv(csv_file)

    # 🧹 DATA CLEANING (fix your error)
    df["Species Name"] = df["Species Name"].fillna("").astype(str)
    df["Description"] = df["Description"].fillna("").astype(str)

    # Remove rows that still have empty descriptions
    df = df[df["Description"].str.strip() != ""]

    print(f"Loaded {len(df)} valid rows")

    results = []

    for i, row in tqdm(df.head(10).iterrows(), total=10):
        name = row["Species Name"]
        ground_truth = row["Description"]
        question = f"Tell me about {name}"

        start = time.time()
        answer = ask_llm(question)
        latency = time.time() - start

        # Extra safety (never crash again)
        answer = str(answer)
        ground_truth = str(ground_truth)
        question = str(question)

        faith = faithfulness_score(answer, ground_truth)
        rel = relevance_score(answer, question)
        hall = hallucination_rate(faith)

        results.append({
            "Question": question,
            "Answer": answer,
            "Faithfulness": round(faith, 3),
            "Relevance": round(rel, 3),
            "Hallucination": round(hall, 3),
            "Latency_sec": round(latency, 2)
        })

    res_df = pd.DataFrame(results)

    print("\n===== AVERAGE SCORES =====")
    print(res_df[["Faithfulness","Relevance","Hallucination","Latency_sec"]].mean())

    res_df.to_csv("phi3_eval_results.csv", index=False)
    print("\nSaved results → phi3_eval_results.csv")


if __name__ == "__main__":
    run_evaluation("claude_descriptions.csv")