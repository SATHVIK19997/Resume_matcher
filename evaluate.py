from src.matcher import rank_resumes
from src.evaluator import evaluate, get_label
from src.utils import load_text, load_resumes, classify, score_bar, SECTION_LABELS

JD_PATH = "data/job_description.txt"
RESUMES_DIR = "data/resumes"

LABEL_MAP = {1.0: "Good Match", 0.5: "Partial Match", 0.0: "Poor Match"}


def main():
    jd_text = load_text(JD_PATH)
    resumes = load_resumes(RESUMES_DIR)

    print(f"\nRunning evaluation on {len(resumes)} resumes...\n")

    ranked = rank_resumes(jd_text, resumes)

    for i, r in enumerate(ranked, 1):
        name = r["name"].replace(".txt", "")
        score = r["score"]
        predicted = classify(score)
        manual = LABEL_MAP.get(get_label(r["name"]), "Unknown")
        match = "OK" if predicted == manual else "MISMATCH"

        print(f"{'-' * 62}")
        print(f"  #{i}  {name}")
        print(f"       Overall Score : {score:.4f}  |  {predicted}  [{match}]  (manual: {manual})")

        if r["breakdown"]:
            print(f"       Section Scores :")
            for section, sec_score in r["breakdown"].items():
                lbl = SECTION_LABELS.get(section, section.title())
                print(f"         {lbl:<12}  {score_bar(sec_score)}  {sec_score:.4f}")

    print(f"{'-' * 62}\n")

    metrics = evaluate(ranked)
    print("--- Metrics ---")
    print(f"Precision@3  : {metrics['precision_at_3']}")
    print(f"Precision@5  : {metrics['precision_at_5']}")
    print(f"nDCG@3       : {metrics['ndcg_at_3']}")
    print(f"nDCG@5       : {metrics['ndcg_at_5']}")
    print(f"Spearman Rho : {metrics['spearman_rho']}")
    print("\nThresholds: >= 0.60 Good | >= 0.50 Partial | < 0.50 Poor")


if __name__ == "__main__":
    main()
