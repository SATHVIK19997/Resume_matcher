import os


def load_text(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def load_resumes(folder):
    resumes = {}
    for filename in os.listdir(folder):
        if filename.endswith(".txt"):
            resumes[filename] = load_text(os.path.join(folder, filename))
    return resumes


SECTION_LABELS = {
    "skills":     "Skills",
    "experience": "Experience",
    "summary":    "Summary",
    "education":  "Education",
}


def classify(score):
    if score >= 0.60:
        return "Good Match"
    elif score >= 0.50:
        return "Partial Match"
    return "Poor Match"


def score_bar(score):
    filled = round(score * 10)
    return "[" + "#" * filled + "-" * (10 - filled) + "]"
