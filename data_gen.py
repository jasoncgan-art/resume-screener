"""Generate the synthetic corpus: job descriptions + resumes with known relevance.

Everything here is SYNTHETIC. No real resumes, no real job postings, no real
names or companies are used or referenced anywhere. Candidate identifiers are
"candidate-NNNN" labels; companies are never named.

World model
-----------
* 12 role families (ML Engineer, Recruiter, ...). Each family owns a pool of
  skills; each skill has 1-3 *surface forms* (e.g. "machine learning" / "ML").
* A job description (JD) lists 7-9 required skills sampled from its family's
  pool (+ a few nice-to-haves, one drawn from another family as noise).
* A resume lists 9-13 skills: ~70% from its own family's pool, ~30% noise
  from other families.
* Surface forms are sampled INDEPENDENTLY for JDs and resumes. So a JD may
  say "ML" while a matching resume says "machine learning" -- the vocabulary
  mismatch that makes naive keyword (ATS-style) screening brittle, and the
  exact signal a semantic (embedding) model can learn from co-occurrence.

Ground truth: a resume is RELEVANT to a JD iff they share the same role
family. Binary labels; the retrieval task is "given a JD, rank the resumes".

Determinism: one `random.Random(SEED)` stream consumed in a fixed order, plus
a seeded numpy Generator. Re-running writes byte-identical files.

Outputs (data/):
    jobs.json       [{job_id, family, title, text}]
    resumes.json    [{candidate_id, family, title, text}]
    split.json      {train_job_ids, test_job_ids}  (stratified 80/20 by family)
    meta.json       {seed, counts, families, sha256 of the three files above}
"""

from __future__ import annotations

import hashlib
import json
import os
import random

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

SEED = 42
N_JOBS_PER_FAMILY = 150
N_RESUMES_PER_FAMILY = 250
TEST_FRAC = 0.20

# --------------------------------------------------------------------------- #
# Skill taxonomy: family -> [(canonical_skill, [surface forms])]
# --------------------------------------------------------------------------- #
# NOTE: surface forms are deliberately varied (abbreviations, alternate
# phrasings). JDs and resumes sample forms independently, which creates the
# vocabulary-mismatch phenomenon keyword search struggles with.
FAMILIES: dict[str, dict] = {
    "ml_engineer": {
        "titles": ["Machine Learning Engineer", "ML Engineer", "Applied ML Engineer"],
        "skills": [
            ("python", ["Python"]),
            ("machine learning", ["machine learning", "ML"]),
            ("deep learning", ["deep learning"]),
            ("pytorch", ["PyTorch"]),
            ("tensorflow", ["TensorFlow"]),
            ("scikit-learn", ["scikit-learn", "sklearn"]),
            ("natural language processing", ["natural language processing", "NLP"]),
            ("computer vision", ["computer vision", "CV"]),
            ("model deployment", ["model deployment", "model serving"]),
            ("mlops", ["MLOps"]),
            ("feature engineering", ["feature engineering"]),
            ("neural networks", ["neural networks"]),
            ("transformers", ["transformers", "transformer models"]),
            ("hyperparameter tuning", ["hyperparameter tuning"]),
            ("a/b testing", ["A/B testing", "experimentation"]),
            ("statistics", ["statistics", "stats"]),
            ("data pipelines", ["data pipelines"]),
            ("docker", ["Docker"]),
            ("large language models", ["large language models", "LLMs"]),
            ("fine-tuning", ["fine-tuning"]),
        ],
    },
    "data_scientist": {
        "titles": ["Data Scientist", "Senior Data Scientist", "Analytics Data Scientist"],
        "skills": [
            ("python", ["Python"]),
            ("statistics", ["statistics", "stats"]),
            ("machine learning", ["machine learning", "ML"]),
            ("data analysis", ["data analysis", "analytics"]),
            ("pandas", ["pandas"]),
            ("sql", ["SQL"]),
            ("data visualization", ["data visualization", "dataviz"]),
            ("hypothesis testing", ["hypothesis testing"]),
            ("regression", ["regression", "regression modeling"]),
            ("a/b testing", ["A/B testing", "experimentation"]),
            ("r", ["R"]),
            ("excel", ["Excel", "spreadsheets"]),
            ("jupyter", ["Jupyter", "Jupyter notebooks"]),
            ("numpy", ["NumPy"]),
            ("scikit-learn", ["scikit-learn", "sklearn"]),
            ("causal inference", ["causal inference"]),
            ("forecasting", ["forecasting", "time series forecasting"]),
            ("dashboards", ["dashboards", "dashboarding"]),
            ("storytelling with data", ["data storytelling", "storytelling with data"]),
            ("matplotlib", ["matplotlib"]),
        ],
    },
    "data_engineer": {
        "titles": ["Data Engineer", "Senior Data Engineer", "Analytics Engineer"],
        "skills": [
            ("python", ["Python"]),
            ("sql", ["SQL"]),
            ("apache spark", ["Apache Spark", "Spark"]),
            ("etl", ["ETL", "extract transform load"]),
            ("data pipelines", ["data pipelines"]),
            ("apache airflow", ["Apache Airflow", "Airflow"]),
            ("data warehousing", ["data warehousing"]),
            ("dbt", ["dbt"]),
            ("apache kafka", ["Apache Kafka", "Kafka"]),
            ("data modeling", ["data modeling"]),
            ("snowflake", ["Snowflake"]),
            ("bigquery", ["BigQuery"]),
            ("docker", ["Docker"]),
            ("aws", ["AWS", "Amazon Web Services"]),
            ("batch processing", ["batch processing"]),
            ("stream processing", ["stream processing", "streaming"]),
            ("data quality", ["data quality"]),
            ("git", ["Git"]),
            ("linux", ["Linux"]),
            ("gcp", ["GCP"]),
        ],
    },
    "backend_engineer": {
        "titles": ["Backend Engineer", "Software Engineer, Backend", "API Engineer"],
        "skills": [
            ("python", ["Python"]),
            ("java", ["Java"]),
            ("go", ["Go", "Golang"]),
            ("rest apis", ["REST APIs", "REST"]),
            ("microservices", ["microservices"]),
            ("sql", ["SQL"]),
            ("postgresql", ["PostgreSQL", "Postgres"]),
            ("redis", ["Redis"]),
            ("docker", ["Docker"]),
            ("kubernetes", ["Kubernetes", "k8s"]),
            ("system design", ["system design"]),
            ("distributed systems", ["distributed systems"]),
            ("message queues", ["message queues"]),
            ("grpc", ["gRPC"]),
            ("unit testing", ["unit testing"]),
            ("ci/cd", ["CI/CD", "continuous integration"]),
            ("linux", ["Linux"]),
            ("git", ["Git"]),
            ("caching", ["caching"]),
            ("graphql", ["GraphQL"]),
        ],
    },
    "frontend_engineer": {
        "titles": ["Frontend Engineer", "Software Engineer, Frontend", "UI Engineer"],
        "skills": [
            ("javascript", ["JavaScript", "JS"]),
            ("typescript", ["TypeScript", "TS"]),
            ("react", ["React", "React.js"]),
            ("html", ["HTML"]),
            ("css", ["CSS"]),
            ("redux", ["Redux"]),
            ("responsive design", ["responsive design"]),
            ("webpack", ["Webpack"]),
            ("rest apis", ["REST APIs", "REST"]),
            ("accessibility", ["accessibility", "a11y"]),
            ("jest", ["Jest", "unit testing"]),
            ("git", ["Git"]),
            ("figma", ["Figma"]),
            ("performance optimization", ["performance optimization", "web performance"]),
            ("graphql", ["GraphQL"]),
            ("ui/ux", ["UI/UX", "user interface design"]),
            ("vue", ["Vue", "Vue.js"]),
            ("npm", ["npm"]),
            ("cross-browser compatibility", ["cross-browser compatibility"]),
            ("state management", ["state management"]),
        ],
    },
    "devops_engineer": {
        "titles": ["DevOps Engineer", "Site Reliability Engineer", "Platform Engineer"],
        "skills": [
            ("linux", ["Linux"]),
            ("docker", ["Docker"]),
            ("kubernetes", ["Kubernetes", "k8s"]),
            ("ci/cd", ["CI/CD", "continuous integration"]),
            ("terraform", ["Terraform"]),
            ("aws", ["AWS", "Amazon Web Services"]),
            ("jenkins", ["Jenkins"]),
            ("prometheus", ["Prometheus", "monitoring"]),
            ("grafana", ["Grafana"]),
            ("bash scripting", ["bash scripting", "shell scripting"]),
            ("ansible", ["Ansible"]),
            ("git", ["Git"]),
            ("cloud infrastructure", ["cloud infrastructure"]),
            ("incident response", ["incident response", "on-call"]),
            ("networking", ["networking"]),
            ("helm", ["Helm"]),
            ("infrastructure as code", ["infrastructure as code", "IaC"]),
            ("load balancing", ["load balancing"]),
            ("logging", ["logging", "centralized logging"]),
            ("security hardening", ["security hardening"]),
        ],
    },
    "hr_generalist": {
        "titles": ["HR Generalist", "Human Resources Generalist", "People Operations Generalist"],
        "skills": [
            ("employee relations", ["employee relations"]),
            ("hr policies", ["HR policies", "policy administration"]),
            ("onboarding", ["onboarding", "new hire onboarding"]),
            ("performance management", ["performance management"]),
            ("benefits administration", ["benefits administration"]),
            ("hris", ["HRIS", "HR systems"]),
            ("employment law", ["employment law"]),
            ("conflict resolution", ["conflict resolution"]),
            ("hr reporting", ["HR reporting", "people analytics"]),
            ("talent management", ["talent management"]),
            ("compliance", ["compliance", "regulatory compliance"]),
            ("excel", ["Excel", "spreadsheets"]),
            ("communication", ["communication", "written communication"]),
            ("hr metrics", ["HR metrics"]),
            ("workforce planning", ["workforce planning"]),
            ("diversity and inclusion", ["diversity and inclusion", "D&I"]),
            ("leave administration", ["leave administration", "LOA administration"]),
            ("offboarding", ["offboarding"]),
            ("job descriptions", ["job descriptions"]),
            ("hr audits", ["HR audits"]),
        ],
    },
    "recruiter": {
        "titles": ["Recruiter", "Technical Recruiter", "Talent Acquisition Specialist"],
        "skills": [
            ("sourcing", ["sourcing", "candidate sourcing"]),
            ("talent acquisition", ["talent acquisition"]),
            ("resume screening", ["resume screening", "candidate screening"]),
            ("interviewing", ["interviewing"]),
            ("boolean search", ["boolean search"]),
            ("linkedin recruiting", ["LinkedIn recruiting", "LinkedIn Recruiter"]),
            ("applicant tracking systems", ["applicant tracking systems", "ATS"]),
            ("offer negotiation", ["offer negotiation"]),
            ("stakeholder management", ["stakeholder management"]),
            ("employer branding", ["employer branding"]),
            ("pipeline management", ["pipeline management"]),
            ("communication", ["communication", "written communication"]),
            ("excel", ["Excel", "spreadsheets"]),
            ("full-cycle recruiting", ["full-cycle recruiting"]),
            ("diversity hiring", ["diversity hiring"]),
            ("reference checks", ["reference checks"]),
            ("onboarding", ["onboarding"]),
            ("salary benchmarking", ["salary benchmarking"]),
            ("campus recruiting", ["campus recruiting"]),
            ("passive candidate outreach", ["passive candidate outreach"]),
        ],
    },
    "financial_analyst": {
        "titles": ["Financial Analyst", "Senior Financial Analyst", "FP&A Analyst"],
        "skills": [
            ("financial modeling", ["financial modeling"]),
            ("excel", ["Excel", "spreadsheets"]),
            ("sql", ["SQL"]),
            ("valuation", ["valuation"]),
            ("budgeting", ["budgeting"]),
            ("forecasting", ["forecasting"]),
            ("variance analysis", ["variance analysis"]),
            ("powerpoint", ["PowerPoint", "presentations"]),
            ("accounting", ["accounting"]),
            ("bloomberg", ["Bloomberg", "Bloomberg Terminal"]),
            ("data analysis", ["data analysis", "analytics"]),
            ("financial reporting", ["financial reporting"]),
            ("vba", ["VBA", "Excel VBA"]),
            ("dcf", ["DCF", "discounted cash flow"]),
            ("pivot tables", ["pivot tables"]),
            ("erp systems", ["ERP systems"]),
            ("month-end close", ["month-end close"]),
            ("kpis", ["KPIs", "key performance indicators"]),
            ("scenario analysis", ["scenario analysis"]),
            ("power bi", ["Power BI"]),
        ],
    },
    "marketing_manager": {
        "titles": ["Marketing Manager", "Digital Marketing Manager", "Growth Marketing Manager"],
        "skills": [
            ("seo", ["SEO", "search engine optimization"]),
            ("content marketing", ["content marketing"]),
            ("email marketing", ["email marketing"]),
            ("google analytics", ["Google Analytics", "GA4"]),
            ("social media marketing", ["social media marketing"]),
            ("campaign management", ["campaign management"]),
            ("hubspot", ["HubSpot", "marketing automation"]),
            ("copywriting", ["copywriting"]),
            ("a/b testing", ["A/B testing", "experimentation"]),
            ("brand strategy", ["brand strategy"]),
            ("ppc", ["PPC", "pay-per-click"]),
            ("communication", ["communication", "written communication"]),
            ("marketing funnels", ["marketing funnels"]),
            ("crm", ["CRM"]),
            ("lead generation", ["lead generation"]),
            ("webinars", ["webinars"]),
            ("customer segmentation", ["customer segmentation"]),
            ("attribution modeling", ["attribution modeling"]),
            ("budget management", ["budget management"]),
            ("cross-functional collaboration", ["cross-functional collaboration"]),
        ],
    },
    "customer_support": {
        "titles": ["Customer Support Specialist", "Customer Success Representative", "Support Agent"],
        "skills": [
            ("customer service", ["customer service"]),
            ("zendesk", ["Zendesk"]),
            ("troubleshooting", ["troubleshooting"]),
            ("ticketing systems", ["ticketing systems"]),
            ("communication", ["communication", "written communication"]),
            ("crm", ["CRM"]),
            ("conflict resolution", ["conflict resolution", "de-escalation"]),
            ("product knowledge", ["product knowledge"]),
            ("live chat support", ["live chat support"]),
            ("documentation", ["documentation"]),
            ("empathy", ["empathy"]),
            ("multitasking", ["multitasking"]),
            ("salesforce", ["Salesforce"]),
            ("phone support", ["phone support"]),
            ("email support", ["email support"]),
            ("customer onboarding", ["customer onboarding"]),
            ("qa testing", ["QA testing", "bug reporting"]),
            ("knowledge base", ["knowledge base"]),
            ("sla management", ["SLA management"]),
            ("customer feedback", ["customer feedback analysis"]),
        ],
    },
    "product_manager": {
        "titles": ["Product Manager", "Senior Product Manager", "Technical Product Manager"],
        "skills": [
            ("product strategy", ["product strategy"]),
            ("roadmapping", ["roadmapping", "product roadmaps"]),
            ("stakeholder management", ["stakeholder management"]),
            ("user research", ["user research"]),
            ("a/b testing", ["A/B testing", "experimentation"]),
            ("sql", ["SQL"]),
            ("data analysis", ["data analysis", "analytics"]),
            ("agile", ["Agile", "Scrum"]),
            ("wireframing", ["wireframing"]),
            ("prioritization", ["prioritization"]),
            ("communication", ["communication", "written communication"]),
            ("jira", ["Jira"]),
            ("product launches", ["product launches", "go-to-market"]),
            ("kpis", ["KPIs", "key performance indicators"]),
            ("prototyping", ["prototyping"]),
            ("competitive analysis", ["competitive analysis"]),
            ("backlog grooming", ["backlog grooming"]),
            ("okrs", ["OKRs"]),
            ("customer interviews", ["customer interviews"]),
            ("metrics definition", ["metrics definition"]),
        ],
    },
}


# Skill implications: canonical skill -> skills it implies.
# A resume listing "PyTorch" implies "machine learning" / "deep learning"
# even when those words never appear -- the inferred-competence phenomenon
# that keyword search cannot capture and semantic models can learn from
# co-occurrence. These edges are part of the synthetic world's ground truth
# (documented here, not hidden) and are used ONLY for relevance labels, never
# as model inputs.
IMPLICATIONS: dict[str, list[str]] = {
    "pytorch": ["machine learning", "deep learning"],
    "tensorflow": ["machine learning", "deep learning"],
    "transformers": ["deep learning", "natural language processing"],
    "large language models": ["natural language processing", "machine learning"],
    "fine-tuning": ["machine learning"],
    "scikit-learn": ["machine learning"],
    "hyperparameter tuning": ["machine learning"],
    "mlops": ["model deployment"],
    "apache airflow": ["data pipelines", "etl"],
    "apache spark": ["data pipelines"],
    "apache kafka": ["stream processing"],
    "dbt": ["data warehousing"],
    "react": ["javascript"],
    "typescript": ["javascript"],
    "vue": ["javascript"],
    "kubernetes": ["docker"],
    "terraform": ["infrastructure as code"],
    "jest": ["unit testing"],
    "boolean search": ["sourcing"],
    "linkedin recruiting": ["sourcing"],
    "full-cycle recruiting": ["talent acquisition", "sourcing", "interviewing"],
    "vba": ["excel"],
    "dcf": ["valuation"],
    "zendesk": ["ticketing systems"],
    "salesforce": ["crm"],
    "jira": ["agile"],
}


def expand_skills(skills: list[str]) -> list[str]:
    """Explicit skills + their implications (deduped, sorted)."""
    out = set(skills)
    for sk in skills:
        out.update(IMPLICATIONS.get(sk, ()))
    return sorted(out)

DEGREES = [
    "B.S. Computer Science", "B.A. Business Administration", "B.S. Engineering",
    "M.S. Data Science", "B.A. Psychology", "MBA", "B.S. Mathematics",
    "B.A. Communications", "B.S. Finance", "M.S. Computer Science",
    "B.A. English", "B.S. Statistics",
]

JD_ABOUT = [
    "We are looking for {article} {title} to join our growing team.",
    "Join our team as {article} {title} and help us build what is next.",
    "We are hiring {article} {title} to take ownership of key initiatives.",
    "An exciting opportunity for {article} {title} to make an immediate impact.",
]

RESUME_SUMMARIES = [
    "{years} {year} of experience as {article} {title} with expertise in {s1}, {s2}, and {s3}.",
    "{title} with {years} {year} of experience specializing in {s1} and {s2}.",
    "Results-driven {title} bringing {years} {year} of experience across {s1}, {s2}, and {s3}.",
]

RESUME_BULLETS = [
    "Hands-on experience with {s1}, {s2}, and {s3} across multiple projects.",
    "Applied {s1} and {s2} to improve team outcomes and delivery quality.",
    "Collaborated with cross-functional stakeholders using {s1} and {s3}.",
    "Recognized for strong {s2} skills and consistent follow-through.",
]


# --------------------------------------------------------------------------- #
# generation
# --------------------------------------------------------------------------- #
# Surface-form realism: JDs are written terse ("ML", "ATS") while resumes
# tend to spell skills out ("machine learning"). This vocabulary mismatch is
# exactly what makes keyword search brittle on real hiring data -- and what a
# semantic model can learn to bridge from co-occurrence. The bias is partial
# (65/35), not total, so keyword search remains a competitive baseline.
FORM_BIAS = 0.65


def _surface(rng: random.Random, skill: tuple[str, list[str]],
             prefer: str | None = None) -> str:
    """Sample one surface form for a skill, independently per occurrence.

    prefer="short" biases toward the tersest form (JD style);
    prefer="long" biases toward the most spelled-out form (resume style);
    None samples uniformly.
    """
    forms = skill[1]
    if prefer is None or len(forms) == 1 or rng.random() >= FORM_BIAS:
        return rng.choice(forms)
    key = min if prefer == "short" else max
    return key(forms, key=len)


def _sample_skills(rng: random.Random, family: str, k: int,
                   exclude: set[str] | None = None) -> list[tuple[str, list[str]]]:
    pool = [s for s in FAMILIES[family]["skills"]
            if exclude is None or s[0] not in exclude]
    return rng.sample(pool, k)


def make_job(rng: random.Random, job_id: int, family: str) -> dict:
    fam = FAMILIES[family]
    title = rng.choice(fam["titles"])
    required = _sample_skills(rng, family, rng.randint(7, 9))
    nice = _sample_skills(rng, family, rng.randint(2, 3),
                          exclude={s[0] for s in required})
    other_fam = rng.choice([f for f in FAMILIES if f != family])
    cross = _sample_skills(rng, other_fam, 1)[0]
    req_forms = [_surface(rng, s, prefer='short') for s in required]
    nice_forms = [_surface(rng, s, prefer='short') for s in nice] + \
        [_surface(rng, cross, prefer='short')]
    article = "an" if title[0].lower() in "aeiou" else "a"
    about = rng.choice(JD_ABOUT).format(title=title, article=article)
    text = (
        f"{title}. {about} "
        f"Required skills: {', '.join(req_forms)}. "
        f"Nice to have: {', '.join(nice_forms)}."
    )
    return {"job_id": job_id, "family": family, "title": title, "text": text,
            "req_skills": sorted(s[0] for s in required)}


def make_resume(rng: random.Random, candidate_id: int, family: str) -> dict:
    fam = FAMILIES[family]
    # 80% headline from own family, 20% adjacent-family title (career switchers)
    if rng.random() < 0.8:
        title = rng.choice(fam["titles"])
    else:
        other = rng.choice([f for f in FAMILIES if f != family])
        title = rng.choice(FAMILIES[other]["titles"])
    years = rng.randint(1, 12)
    n_own = rng.randint(6, 9)
    n_noise = rng.randint(3, 4)
    own = _sample_skills(rng, family, n_own)
    noise: list[tuple[str, list[str]]] = []
    for _ in range(n_noise):
        of = rng.choice([f for f in FAMILIES if f != family])
        noise.append(_sample_skills(rng, of, 1)[0])
    skills = own + noise
    rng.shuffle(skills)
    skill_forms = [_surface(rng, s, prefer='long') for s in skills]
    s1, s2, s3 = (skill_forms + skill_forms)[:3]
    article = "an" if title[0].lower() in "aeiou" else "a"
    yr = "year" if years == 1 else "years"
    summary = rng.choice(RESUME_SUMMARIES).format(
        years=years, year=yr, article=article, title=title, s1=s1, s2=s2, s3=s3)
    bullets = []
    for tmpl in rng.sample(RESUME_BULLETS, 2):
        pool = rng.sample(skill_forms, 3)
        bullets.append(tmpl.format(s1=pool[0], s2=pool[1], s3=pool[2]))
    degree = rng.choice(DEGREES)
    text = (
        f"Candidate {candidate_id:04d} - {title}. {summary} "
        f"Skills: {', '.join(skill_forms)}. "
        f"Experience: {' '.join(bullets)} "
        f"Education: {degree}."
    )
    canon = sorted(s[0] for s in skills)
    return {"candidate_id": candidate_id, "family": family,
            "title": title, "text": text,
            "skills": canon,
            "skills_expanded": expand_skills(canon)}


def generate(seed: int = SEED) -> tuple[list[dict], list[dict], dict]:
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)  # reserved for future numeric fields
    jobs, resumes = [], []
    job_id = 0
    for family in FAMILIES:
        for _ in range(N_JOBS_PER_FAMILY):
            jobs.append(make_job(rng, job_id, family))
            job_id += 1
    cid = 0
    for family in FAMILIES:
        for _ in range(N_RESUMES_PER_FAMILY):
            resumes.append(make_resume(rng, cid, family))
            cid += 1
    # stratified 80/20 JD split (resumes are the fixed retrieval corpus)
    train_ids, test_ids = [], []
    for family in FAMILIES:
        fam_ids = [j["job_id"] for j in jobs if j["family"] == family]
        rng.shuffle(fam_ids)
        n_test = int(len(fam_ids) * TEST_FRAC)
        test_ids.extend(fam_ids[:n_test])
        train_ids.extend(fam_ids[n_test:])
    rng.shuffle(train_ids)
    rng.shuffle(test_ids)
    split = {"train_job_ids": sorted(train_ids), "test_job_ids": sorted(test_ids)}
    return jobs, resumes, split


def main() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    jobs, resumes, split = generate(SEED)
    payloads = {
        "jobs.json": jobs,
        "resumes.json": resumes,
        "split.json": split,
    }
    hashes = {}
    for name, obj in payloads.items():
        path = os.path.join(DATA_DIR, name)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=1)
        with open(path, "rb") as f:
            hashes[name] = hashlib.sha256(f.read()).hexdigest()
    meta = {
        "seed": SEED,
        "n_families": len(FAMILIES),
        "families": sorted(FAMILIES),
        "n_jobs": len(jobs),
        "n_resumes": len(resumes),
        "n_train_jobs": len(split["train_job_ids"]),
        "n_test_jobs": len(split["test_job_ids"]),
        "sha256": hashes,
        "disclaimer": "All job descriptions and resumes are SYNTHETIC, generated "
                      "from templates with a fixed random seed. No real people, "
                      "companies, or resumes are used or referenced.",
    }
    with open(os.path.join(DATA_DIR, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    print(f"[data_gen] jobs={len(jobs)} resumes={len(resumes)} "
          f"train_jds={len(split['train_job_ids'])} test_jds={len(split['test_job_ids'])} "
          f"-> {DATA_DIR}")


if __name__ == "__main__":
    main()
