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
│   ├── preprocessor.py       # text cleaning
│   ├── matcher.py            # the core scoring logic
│   └── evaluator.py          # metrics calculation
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
jupyter notebook notebook/evaluation.ipynb
```

---

## Technical approach

I used `sentence-transformers` with the `all-MiniLM-L6-v2` model. The way it works — both the JD and each resume get encoded into embedding vectors, and then I compute cosine similarity between them. Output is a score from 0 to 1 per resume, which I use to rank them.

I considered TF-IDF as a simpler baseline. TF-IDF is fast and easy to set up, but it only looks at word frequency — it completely misses semantic meaning. So if a candidate says "prompt engineering and LLM fine-tuning" but the JD says "optimize model outputs", TF-IDF would give that a low score even though it's clearly relevant. Sentence transformers handle this naturally.

**Preprocessing steps:**
- lowercase everything
- remove special characters and punctuation
- normalize whitespace
- stopword removal is available but off by default — the model works better with full sentences

---

## Evaluation

Since there's no labeled dataset available, I created a synthetic one — a job description for a Senior AI Applications Engineer role and 10 resumes covering a range of profiles from strong matches to completely irrelevant ones. I manually labeled each:

| Resume | Label |
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
- nDCG@5: **1.0** — ranking order was correct
- Spearman Rho: **0.905** — strong correlation with manual labels

**If I had a larger labeled dataset**, the metrics I'd focus on are:
- **nDCG@K** — most important for ranking tasks, it rewards putting better candidates higher not just getting the binary hit/miss
- **Precision@K** — practical for recruiters who only look at top N results
- **Recall@K** — makes sure we're not missing good candidates
- **Spearman Rho** — good summary metric to track overall alignment with human judgment

---

## Known limitations

- The model is general purpose — it hasn't been fine-tuned on HR or recruiting data specifically
- It treats the whole resume as one chunk of text, so skills and experience sections aren't weighted any differently than the address line
- The classification thresholds (0.75 for Good, 0.55 for Partial) were picked manually based on what looked right — ideally these get calibrated on real labeled data
- No structured field extraction — years of experience, location, specific certifications aren't factored in separately

---

## What I'd do next

- Parse resumes into sections (Skills, Experience, Education) and weight them differently in the score
- Add a hard filter layer on top — things like minimum years of experience, required skills checklist
- Fine-tune the embedding model on actual recruiting data if we can get labeled hire/no-hire decisions
- Wrap it in a FastAPI service, containerize with Docker, deploy on AWS Lambda
- Build a feedback loop so recruiters can flag bad rankings and we retrain over time
