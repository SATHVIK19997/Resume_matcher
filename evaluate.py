from src.matcher import rank_resumes
from src.evaluator import evaluate, get_label
from src.utils import load_text, load_resumes

JD_PATH = "data/job_description.txt"
RESUMES_DIR = "data/resumes"

LABEL_MAP = {1.0: "Good Match", 0.5: "Partial Match", 0.0: "Poor Match"}


def main():
    jd_text = load_text(JD_PATH)
    resumes = load_resumes(RESUMES_DIR)

    print(f"\nRunning evaluation on {len(resumes)} resumes...\n")

    ranked = rank_resumes(jd_text, resumes)

    print(f"{'Rank':<5} {'Resume':<38} {'Score':<8} {'Predicted':<18} {'Manual Label'}")
    print("-" * 85)

    for i, r in enumerate(ranked, 1):
        name = r["name"].replace(".txt", "")
        score = r["score"]
        predicted = "Good Match" if score >= 0.60 else ("Partial Match" if score >= 0.50 else "Poor Match")
        manual = LABEL_MAP.get(get_label(r["name"]), "Unknown")
        print(f"{i:<5} {name:<38} {score:<8.4f} {predicted:<18} {manual}")

    metrics = evaluate(ranked)
    print("\n--- Metrics ---")
    print(f"Precision@3  : {metrics['precision_at_3']}")
    print(f"Precision@5  : {metrics['precision_at_5']}")
    print(f"nDCG@3       : {metrics['ndcg_at_3']}")
    print(f"nDCG@5       : {metrics['ndcg_at_5']}")
    print(f"Spearman Rho : {metrics['spearman_rho']}")
    print("\nThresholds: >= 0.60 Good | >= 0.50 Partial | < 0.50 Poor")


if __name__ == "__main__":
    main()
