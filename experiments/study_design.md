# HARE Human Evaluation Study Design

## 1. Objective

Validate that HARE's generative recommendation produces outputs that are
(a) higher quality than baseline methods, (b) more personalized across users,
and (c) increasingly specific over repeated interactions -- as measured by
human raters.

---

## 2. Study Overview

| Parameter           | Value                                             |
|---------------------|---------------------------------------------------|
| Design              | Within-subjects, blinded A/B comparison           |
| Primary domain      | Claude Skills (structured markdown)               |
| Secondary domain    | Study plan generation                             |
| Participants        | 20-30 (technical background, familiar with LLMs)  |
| Sessions per user   | 10 interaction rounds per method                  |
| Methods compared    | HARE, RAG baseline, Vanilla GPT-2, Random         |
| Presentation order  | Counterbalanced (Latin square)                    |

---

## 3. Evaluation Dimensions

Each generated output is rated on a 5-point Likert scale:

### 3.1 Quality (1-5)
> "How well-written and coherent is this output?"
- 1 = Incoherent, broken formatting, nonsensical
- 3 = Readable but generic, template-like
- 5 = Professional quality, well-structured, immediately usable

### 3.2 Relevance (1-5)
> "How relevant is this output to your stated need?"
- 1 = Completely irrelevant, wrong topic
- 3 = Somewhat relevant but misses key aspects
- 5 = Directly addresses exactly what I need

### 3.3 Personalization (1-5)
> "How well does this output reflect your specific context and preferences?"
- 1 = Generic, could be for anyone
- 3 = Some customization but largely generic
- 5 = Clearly tailored to my specific situation and needs

### 3.4 Novelty (1-5)
> "Does this output offer something beyond what you could easily find or generate yourself?"
- 1 = Nothing new, trivially available elsewhere
- 3 = Some useful combinations but mostly derivative
- 5 = Genuinely novel synthesis that I would not have found on my own

### 3.5 Pairwise Preference
> "Which output do you prefer overall?" (forced choice between HARE vs each baseline)

---

## 4. Protocol

### Phase 1: Intake (5 min)
- Collect demographic info (role, years of experience, LLM familiarity)
- Participant selects 3 topics of interest from a predefined list
  (e.g., "Python testing", "API security", "data pipelines")
- This establishes the "user context" for personalization

### Phase 2: Baseline Round (10 min)
- Participant interacts with each method for 10 rounds on the same topics
- Each round: participant sees a generated skill, rates it on all 4 dimensions
- After rating, participant provides binary feedback (useful/not useful)
  which feeds back into HARE's user model (but not into baselines)
- Methods are presented in counterbalanced order, unlabeled ("System A/B/C/D")

### Phase 3: Comparison Round (5 min)
- Side-by-side display of HARE vs each baseline output for the same query
- Forced-choice preference (which is better?)
- Optional free-text explanation of preference

### Phase 4: Post-Study Survey (5 min)
- Overall preference ranking across all methods
- Free-text feedback on what made outputs feel personalized vs generic
- Self-reported engagement trajectory ("Did outputs get better over time?")

---

## 5. Hypotheses

**H1 (Quality):** HARE mean quality >= RAG baseline mean quality (non-inferiority).
HARE should not sacrifice quality for personalization.

**H2 (Personalization):** HARE mean personalization > all baselines (p < 0.05).
The core claim: user modeling produces more personalized outputs.

**H3 (Personalization trajectory):** HARE personalization ratings increase
over rounds (positive slope in linear regression of personalization vs round).
The system learns and improves.

**H4 (Preference):** HARE preferred over each baseline in pairwise comparison
(> 50% preference rate, binomial test p < 0.05).

**H5 (Novelty):** HARE mean novelty > RAG and Vanilla GPT-2 baselines.
Synthesis produces genuinely novel outputs that retrieval cannot.

---

## 6. Statistical Analysis Plan

### Primary analysis
- Mixed-effects linear model with participant as random effect:
  `rating ~ method + round + method*round + (1|participant)`
- The method*round interaction tests whether HARE improves over rounds
  while baselines remain flat (H3)

### Pairwise comparisons
- Wilcoxon signed-rank test for each dimension (HARE vs each baseline)
- Bonferroni correction for multiple comparisons (3 baselines x 4 dimensions)

### Effect sizes
- Cohen's d for each pairwise comparison
- Report confidence intervals, not just p-values

### Personalization divergence (automated)
- For each query posed to multiple users: compute pairwise cosine distance
  of generated outputs
- HARE should show significantly higher divergence than RAG (which produces
  identical outputs for identical queries)

---

## 7. Recruitment & IRB Considerations

### Participants
- Recruit from: university CS/DS programs, developer communities, Discord
- Inclusion: familiar with LLMs, can evaluate technical writing quality
- Compensation: $15 gift card or course credit (30-min study)

### IRB
- This study involves human evaluation of AI-generated text
- No deception, no sensitive data collected
- Likely qualifies for IRB exemption (educational/quality evaluation)
- Required documentation:
  - Informed consent form
  - Data handling plan (anonymized ratings, no PII stored)
  - Study protocol for reproducibility

### Ethical considerations
- Generated outputs are evaluated, not deployed -- no risk of harm from
  low-quality generations
- Participants are rating text quality, not being manipulated
- All methods are disclosed post-study (debriefing)

---

## 8. Data Collection Infrastructure

### Technical setup
- Web interface showing generated outputs (Flask/Streamlit app)
- Backend runs all 4 methods on each query, presents outputs unlabeled
- Ratings collected via Likert scale buttons
- Session data logged: query, method, generated output, all ratings, timestamps

### Reproducibility
- Fix random seeds for all methods
- Log all model checkpoints used
- Store raw generated outputs alongside ratings
- Pre-register hypotheses and analysis plan (OSF)

---

## 9. Expected Outcomes & Power Analysis

### Minimum detectable effect
- With N=20 participants, 10 rounds each, 4 methods:
  800 total ratings per dimension
- For paired t-test at alpha=0.05, power=0.80:
  can detect Cohen's d >= 0.45 (medium effect)
- For the interaction effect (learning trajectory):
  can detect a slope difference of ~0.1 points per round

### What "success" looks like
- HARE personalization mean >= 3.5 (above midpoint)
- HARE preferred over RAG in >= 60% of pairwise comparisons
- Statistically significant positive slope for HARE personalization over rounds
- HARE novelty ratings significantly higher than all retrieval-based baselines

### What "partial success" looks like
- HARE preferred but quality not significantly better (shows personalization
  value without quality loss -- still publishable)
- Learning trajectory visible but not statistically significant at N=20
  (suggests larger study needed -- still publishable with N=20 pilot framing)

---

## 10. Timeline

| Week | Activity                                    |
|------|---------------------------------------------|
| 1    | Build evaluation web interface              |
| 2    | Pilot study (3-5 participants), refine UI   |
| 3-4  | Full study recruitment and data collection  |
| 5    | Analysis and write-up                       |
