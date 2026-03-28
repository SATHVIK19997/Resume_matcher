import argparse
from src.matcher import rank_resumes
from src.utils import load_text, load_resumes

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


def section_bar(score):
    filled = round(score * 10)
    return "[" + "#" * filled + "-" * (10 - filled) + "]"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jd", required=True, help="path to job description file")
    parser.add_argument("--resumes", required=True, help="path to folder with resume files")
    args = parser.parse_args()

    jd_text = load_text(args.jd)
    resumes = load_resumes(args.resumes)

    print(f"\nScoring {len(resumes)} resumes against JD...\n")
    ranked = rank_resumes(jd_text, resumes)

    for i, r in enumerate(ranked, 1):
        name = r["name"].replace(".txt", "")
        score = r["score"]
        label = classify(score)

        print(f"{'-' * 62}")
        print(f"  #{i}  {name}")
        print(f"       Overall Score : {score:.4f}  |  {label}")

        if r["breakdown"]:
            print(f"       Section Scores :")
            for section, sec_score in r["breakdown"].items():
                bar = section_bar(sec_score)
                lbl = SECTION_LABELS.get(section, section.title())
                print(f"         {lbl:<12}  {bar}  {sec_score:.4f}")

    print(f"{'-' * 62}")


if __name__ == "__main__":
    main()
