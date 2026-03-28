import numpy as np
from scipy.stats import spearmanr

# manually labeled scores for the test resumes
# 1.0 = strong fit, 0.5 = partial fit, 0.0 = not a fit
GROUND_TRUTH = {
    "resume_01_arjun_sharma":      1.0,
    "resume_02_priya_nair":        1.0,
    "resume_03_rahul_verma":       1.0,
    "resume_04_sneha_kulkarni":    0.5,
    "resume_05_kiran_reddy":       0.5,
    "resume_06_aditya_joshi":      0.5,
    "resume_07_meera_thomas":      0.0,
    "resume_08_vijay_kumar":       0.0,
    "resume_09_pooja_mehta":       0.0,
    "resume_10_sathvik_kantharaj": 1.0,
}


def get_label(resume_name):
    # strip the path and extension to get just the key
    key = resume_name.replace(".txt", "").split("/")[-1].split("\\")[-1]
    return GROUND_TRUTH.get(key, -1.0)


def precision_at_k(ranked_results, k, threshold=0.75):
    top_k = ranked_results[:k]
    hits = sum(1 for r in top_k if get_label(r["name"]) >= threshold)
    return round(hits / k, 4)


def ndcg_at_k(ranked_results, k):
    top_k = ranked_results[:k]
    labels = [get_label(r["name"]) for r in top_k]

    dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(labels))

    # what the perfect ranking would look like
    ideal_labels = sorted(GROUND_TRUTH.values(), reverse=True)[:k]
    idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_labels))

    return round(dcg / idcg, 4) if idcg > 0 else 0.0


def spearman_correlation(ranked_results):
    predicted = [r["score"] for r in ranked_results]
    actual = [get_label(r["name"]) for r in ranked_results]

    pairs = [(p, a) for p, a in zip(predicted, actual) if a >= 0]
    if len(pairs) < 2:
        return 0.0

    pred_scores, true_labels = zip(*pairs)
    rho, _ = spearmanr(pred_scores, true_labels)
    return round(float(rho), 4)


def evaluate(ranked_results):
    return {
        "precision_at_3": precision_at_k(ranked_results, k=3),
        "precision_at_5": precision_at_k(ranked_results, k=5),
        "ndcg_at_3":      ndcg_at_k(ranked_results, k=3),
        "ndcg_at_5":      ndcg_at_k(ranked_results, k=5),
        "spearman_rho":   spearman_correlation(ranked_results),
    }
