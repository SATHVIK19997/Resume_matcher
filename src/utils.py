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
