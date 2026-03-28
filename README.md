# Resume Matcher

This is my submission for the AI Resume Matching take-home assignment. The problem is straightforward — manual resume screening is slow, inconsistent, and good candidates often get missed. The ask was to build a system that takes a job description and a set of resumes and produces a relevance score for each one.

Instead of going with basic keyword matching, I used sentence embeddings so the system understands meaning rather than just looking for exact word matches. For example, "built LLM-powered agent workflows" and "developed agentic AI systems" should score similarly even though the words are different — a keyword-based approach would miss that entirely.

---

## What's inside

```
Resume_matcher/
├── data/
│   ├── job_description.txt
│   └── resumes/              # 10 sample resumes for testing
├── src/
│   ├── preprocessor.py       # text cleaning, section parsing, weights
│   ├── matcher.py            # core scoring logic (section-aware)
│   ├── evaluator.py          # metrics calculation
│   └── utils.py              # shared file loading helpers
├── notebook/
│   └── evaluation.ipynb      # visualizations and analysis
├── main.py                   # run this to score your own resumes
├── evaluate.py               # runs evaluation against labeled test set
└── requirements.txt
```

---

## Getting started

Clone and install dependencies:

```bash
git clone https://github.com/SATHVIK19997/Resume_matcher.git
cd Resume_matcher
pip install -r requirements.txt
```

---

## How to use it

**Score resumes against a JD:**
```bash
python main.py --jd data/job_description.txt --resumes data/resumes/
```

**Run the evaluation (with metrics):**
```bash
python evaluate.py
```

**Open the notebook:**
```bash
python -m notebook notebook/evaluation.ipynb
```

---

## Technical approach

### Model — sentence-transformers (`all-MiniLM-L6-v2`)

Each resume is split into sections (Skills, Experience, Summary, Education) and each section is encoded separately into a 384-dimensional embedding vector using `all-MiniLM-L6-v2`. The same is done for the job description. Cosine similarity is then computed between the JD vector and each section vector, and the results are combined into a final weighted score.

I considered TF-IDF as a simpler baseline. TF-IDF is fast and easy to set up, but it only looks at word frequency — it completely misses semantic meaning. If a candidate says "prompt engineering and LLM fine-tuning" but the JD says "optimize model outputs", TF-IDF would give that a low score even though it's clearly relevant. Sentence transformers handle this naturally.

### Section-aware scoring

Rather than treating the entire resume as one block of text, the system parses each resume into named sections and scores them independently. The final score is a weighted average:

| Section | Weight | Reason |
|---|---|---|
| Technical Skills | 40% | Most directly signals role fit |
| Work Experience | 40% | Validates practical application |
| Summary / Profile | 15% | Useful context but easier to game |
| Education | 5% | Least relevant for senior engineering roles |

If a section is missing from a resume, its weight is redistributed proportionally to the sections that are present. If no sections are detected at all, the system falls back to scoring the full resume text.

### Preprocessing steps

- lowercase everything
- remove special characters and punctuation
- normalize whitespace
- stopword removal is available but off by default — the model works better with full natural language sentences

### Score classification thresholds

| Score | Label |
|---|---|
| >= 0.60 | Good Match |
| >= 0.50 | Partial Match |
| < 0.50 | Poor Match |

These thresholds were calibrated against the section-aware scoring distribution. With whole-text scoring the scores were higher (previously 0.75/0.55), but section-level encoding produces lower absolute similarity values, so the thresholds were adjusted accordingly.

---

## Evaluation

Since there's no labeled dataset available, I created a synthetic one — a job description for a Senior AI Applications Engineer role at Ema and 10 resumes covering a range of profiles from strong matches to completely irrelevant ones. I manually labeled each before running the model:

| Resume | Manual Label |
|---|---|
| Arjun Sharma | Good Match |
| Priya Nair | Good Match |
| Rahul Verma | Good Match |
| Sathvik Kantharaj | Good Match |
| Sneha Kulkarni | Partial Match |
| Kiran Reddy | Partial Match |
| Aditya Joshi | Partial Match |
| Meera Thomas | Poor Match |
| Vijay Kumar | Poor Match |
| Pooja Mehta | Poor Match |

Results after running `evaluate.py`:
- Precision@3: **1.0** — top 3 results were all actual good matches
- nDCG@5: **1.0** — ranking order was correct for top 5
- Spearman Rho: **0.905** — strong correlation with manual labels
- 9 out of 10 predicted correctly

The one miss: Kiran Reddy (data scientist with Python/SQL but no LLM experience) scored just below the Partial threshold. His skills section didn't match the AI-specific language in the JD closely enough. This is a known edge case — a hard filter on required skills would catch it.

**If I had a larger labeled dataset**, the metrics I'd focus on are:
- **nDCG@K** — most important for ranking tasks, it rewards putting better candidates higher not just getting the binary hit/miss right
- **Precision@K** — practical for recruiters who only look at top N results
- **Recall@K** — makes sure we're not missing good candidates
- **Spearman Rho** — good summary metric to track overall alignment with human judgment

---

## Known limitations

- The model is general purpose — it hasn't been fine-tuned on HR or recruiting data specifically
- Section weights (40/40/15/5) and score thresholds (0.60/0.50) were set manually — they need calibration on real labeled data
- No hard filters — minimum years of experience, required certifications, and location aren't enforced separately
- Currently works on plain text files only — PDF and DOCX support isn't built yet

---

## What I'd do next

- Add PDF/DOCX parsing — extract structured fields like current title, years of experience, location
- Add a hard filter layer on top of the semantic score — minimum experience, required skills checklist
- Fine-tune the embedding model on actual recruiting data if we can get labeled hire/no-hire decisions
- Wrap it in a FastAPI service, containerize with Docker, deploy on AWS Lambda
- Build a feedback loop so recruiters can flag bad rankings and we retrain over time
