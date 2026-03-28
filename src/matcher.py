from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from src.preprocessor import preprocess, parse_sections, get_weights_for_sections

# loading at module level so we don't reload the model on every call
model = SentenceTransformer("all-MiniLM-L6-v2")


def _cosine(vec_a, vec_b):
    return float(cosine_similarity(vec_a, vec_b)[0][0])


def score_resume(jd_text, resume_text):
    """
    Section-aware scoring. Returns overall score + per-section breakdown.
    Falls back to whole-text scoring if no sections are detected.
    """
    jd = preprocess(jd_text)
    jd_vec = model.encode([jd])

    sections = parse_sections(resume_text)

    if not sections:
        resume_vec = model.encode([preprocess(resume_text)])
        score = round(_cosine(jd_vec, resume_vec), 4)
        return {"score": score, "breakdown": {}}

    weights = get_weights_for_sections(sections)

    total_score = 0.0
    breakdown = {}

    for section, text in sections.items():
        section_vec = model.encode([preprocess(text)])
        sim = round(_cosine(jd_vec, section_vec), 4)
        breakdown[section] = sim
        total_score += sim * weights[section]

    return {"score": round(total_score, 4), "breakdown": breakdown}


def rank_resumes(jd_text, resumes):
    # resumes is a dict: {filename: text}
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
