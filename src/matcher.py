from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from src.preprocessor import preprocess, parse_sections, get_weights_for_sections

# load once at module level — no point reloading on every call
model = SentenceTransformer("all-MiniLM-L6-v2")


def _cosine(a, b):
    return float(cosine_similarity(a, b)[0][0])


def score_resume(jd_text, resume_text):
    jd = preprocess(jd_text)
    jd_vec = model.encode([jd])

    sections = parse_sections(resume_text)

    if not sections:
        # no sections detected, fall back to scoring the full text
        score = round(_cosine(jd_vec, model.encode([preprocess(resume_text)])), 4)
        return {"score": score, "breakdown": {}}

    weights = get_weights_for_sections(sections)
    total = 0.0
    breakdown = {}

    for section, text in sections.items():
        sim = round(_cosine(jd_vec, model.encode([preprocess(text)])), 4)
        breakdown[section] = sim
        total += sim * weights[section]

    return {"score": round(total, 4), "breakdown": breakdown}


def rank_resumes(jd_text, resumes):
    results = []
    for name, text in resumes.items():
        result = score_resume(jd_text, text)
        results.append({
            "name":      name,
            "score":     result["score"],
            "breakdown": result["breakdown"],
        })
    results.sort(key=lambda x: x["score"], reverse=True)
    return results
