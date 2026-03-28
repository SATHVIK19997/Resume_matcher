import re
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)

STOP_WORDS = set(stopwords.words("english"))

# skills and experience carry the most signal for technical roles
SECTION_WEIGHTS = {
    "skills":     0.40,
    "experience": 0.40,
    "summary":    0.15,
    "education":  0.05,
}

SECTION_HEADERS = {
    "skills":     ["technical skills", "skills", "core skills", "key skills", "tech stack"],
    "experience": ["work experience", "employment history", "experience", "professional experience"],
    "summary":    ["summary", "profile", "about", "objective", "overview"],
    "education":  ["education", "academic", "qualification", "degree"],
}


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def remove_stopwords(text):
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOP_WORDS]
    return " ".join(tokens)


def preprocess(text, remove_stops=False):
    # sentence-transformers work better with full sentences so stopwords are off by default
    text = clean_text(text)
    if remove_stops:
        text = remove_stopwords(text)
    return text


def parse_sections(resume_text):
    lines = resume_text.splitlines()
    sections = {}
    current_section = None
    current_lines = []

    def flush(section, lines):
        text = " ".join(lines).strip()
        if text:
            sections[section] = text

    for line in lines:
        stripped = line.strip().lower().rstrip(":")
        matched = None
        for section, keywords in SECTION_HEADERS.items():
            if stripped in keywords:
                matched = section
                break

        if matched:
            if current_section:
                flush(current_section, current_lines)
            current_section = matched
            current_lines = []
        elif current_section:
            current_lines.append(line.strip())

    if current_section:
        flush(current_section, current_lines)

    return sections


def get_weights_for_sections(found_sections):
    # if a section is missing just redistribute its weight to whatever is present
    available = {k: v for k, v in SECTION_WEIGHTS.items() if k in found_sections}
    if not available:
        return {}
    total = sum(available.values())
    return {k: round(v / total, 4) for k, v in available.items()}
