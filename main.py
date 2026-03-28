import argparse
from src.matcher import rank_resumes
from src.utils import load_text, load_resumes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jd", required=True, help="path to job description file")
    parser.add_argument("--resumes", required=True, help="path to folder with resume files")
    args = parser.parse_args()

    jd_text = load_text(args.jd)
    resumes = load_resumes(args.resumes)

    print(f"\nScoring {len(resumes)} resumes against JD...\n")
    ranked = rank_resumes(jd_text, resumes)

    print(f"{'Rank':<6} {'Resume':<45} {'Score'}")
    print("-" * 62)
    for i, r in enumerate(ranked, 1):
        print(f"{i:<6} {r['name']:<45} {r['score']:.4f}")


if __name__ == "__main__":
    main()
