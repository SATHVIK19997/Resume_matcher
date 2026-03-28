from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from src.preprocessor import preprocess, parse_sections, get_weights_for_sections

# loading at module level so we don't reload the model on every call
model = SentenceTransformer("all-MiniLM-L6-v2")


def _cosine(vec_a, vec_b):
    return float(cosine_similarity(vec_a, vec_b)[0][0])


def score_resume(jd_text, resume_text):
    """
    Section-aware scoring. Each section (skills, experience, summary, education)
    is scored against the full JD separately, then combined using weighted average.
    Falls back to whole-text scoring if no sections are detected.
    """
    jd = preprocess(jd_text)
    jd_vec = model.encode([jd])

    sections = parse_sections(resume_text)

    if not sections:
        # no sections found, fall back to scoring the whole text
        resume_vec = model.encode([preprocess(resume_text)])
        return round(_cosine(jd_vec, resume_vec), 4)

    weights = get_weights_for_sections(sections)

    total_score = 0.0
    for section, text in sections.items():
        section_vec = model.encode([preprocess(text)])
        sim = _cosine(jd_vec, section_vec)
        total_score += sim * weights[section]

    return round(total_score, 4)


def rank_resumes(jd_text, resumes):
    # resumes is a dict: {filename: text}
    results = []
    for name, text in resumes.items():
        score = score_resume(jd_text, text)
        results.append({"name": name, "score": score})

    results.sort(key=lambda x: x["score"], reverse=True)
    return results
