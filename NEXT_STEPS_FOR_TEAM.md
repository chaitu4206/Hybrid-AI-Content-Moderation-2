# Next Steps for Other Team Members

## Current Starting Point

The five BERT classifiers and the unified inference layer are already implemented.

Do NOT start by retraining all five models.

The immediate task is to build the components after unified inference.

---

# Step 1 — Understand the Unified Output

Run:

```powershell
cd C:\Users\chait\unified_moderation
python unified_inference.py
```

Enter several representative examples.

Record the outputs from:
- Toxicity
- Sensitive Content
- Hate Speech
- Spam
- Violence

The next modules should consume these numerical outputs rather than re-run unrelated preprocessing.

---

# Step 2 — Build a Unified Feature Extractor

Create:

```text
feature_extractor.py
```

Its purpose is to convert the five model outputs into one numerical feature vector.

Example:

```text
[
    toxicity_1,
    toxicity_2,
    toxicity_3,
    toxicity_4,
    toxicity_5,
    toxicity_6,
    sensitive_score,
    hate_score,
    offensive_score,
    spam_score,
    violence_score
]
```

The exact toxicity feature mapping must use the verified label meanings from the toxicity training configuration.

---

# Step 3 — Create a Severity Dataset

Create a dataset containing:

```text
model features -> severity label
```

Example structure:

```csv
toxicity_1,toxicity_2,...,violence_score,severity
0.01,0.02,...,0.05,LOW
0.45,0.10,...,0.60,MEDIUM
0.80,0.70,...,0.90,HIGH
```

Do not invent final severity labels without documenting how they were obtained.

A reasonable project method is to define labels using documented rules or expert annotation, then train/evaluate the Random Forest on those labels.

---

# Step 4 — Train Random Forest

Create:

```text
severity_model.py
```

Use:

```python
from sklearn.ensemble import RandomForestClassifier
```

Train on the feature dataset.

Save the trained model:

```text
models/severity_random_forest.pkl
```

Evaluate it with:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

---

# Step 5 — Build the Decision Engine

Create:

```text
decision_engine.py
```

Input:

```text
BERT outputs
+
severity
```

Output:

```text
ALLOW
WARN
REMOVE
ESCALATE
```

Keep the rules explicit.

Example structure:

```python
def make_decision(severity, model_results):

    # documented project rules go here

    return action
```

The team should decide and document the actual rules.

---

# Step 6 — Integrate Everything

Modify or create:

```text
hybrid_pipeline.py
```

The flow should be:

```text
Input Text
    |
    v
Five BERT Models
    |
    v
Feature Extraction
    |
    v
Random Forest Severity
    |
    v
Decision Engine
    |
    v
Final Action
```

Expected output:

```text
Content:
...

Toxicity:
...

Sensitive:
...

Hate Speech:
...

Spam:
...

Violence:
...

Severity:
HIGH

Recommended Action:
ESCALATE
```

---

# Step 7 — Human-in-the-Loop

Create a reviewer interface.

The reviewer should see:

```text
Original content
Model predictions
Probabilities
Severity
Recommended action
```

Then allow:

```text
APPROVE
REJECT
ESCALATE
```

Store reviewer decisions for later evaluation.

---

# Step 8 — Moderator Dashboard

Create:

```text
app.py
```

Use Streamlit.

Install if necessary:

```powershell
pip install streamlit
```

Run:

```powershell
streamlit run app.py
```

Dashboard sections:

```text
CONTENT INPUT
       |
       v
AI ANALYSIS
       |
       +-- Toxicity
       +-- Sensitive
       +-- Hate Speech
       +-- Spam
       +-- Violence
       |
       v
SEVERITY
       |
       v
RECOMMENDED ACTION
       |
       v
HUMAN REVIEW
```

---

# Step 9 — Test the Complete System

Prepare test cases covering:

1. Normal/benign content
2. Toxic content
3. Sensitive content
4. Hate speech
5. Spam
6. Violent content
7. Content with multiple categories
8. Ambiguous content

For every test record:

```text
Input
Model outputs
Severity
Decision
Human decision
```

---

# Step 10 — End-to-End Evaluation

Calculate:

### Classifier level
- Precision
- Recall
- F1
- Accuracy where appropriate

### Severity model
- Accuracy
- Precision
- Recall
- F1

### Complete system
- Correct decisions
- False positives
- False negatives
- Escalation rate
- Human override rate

---

# Step 11 — Documentation

Update the README with:

- Architecture
- Dataset information
- Model information
- Training procedure
- Metrics
- Severity model
- Decision rules
- Dashboard
- Screenshots
- Limitations
- Future work

---

# Step 12 — Final Project Paper

The final paper should contain:

```text
Abstract
Introduction
Problem Statement
Objectives
Literature Survey
Datasets
Preprocessing
Methodology
Five BERT Models
Unified Inference
Random Forest Severity Scoring
Decision Engine
Human-in-the-Loop
Dashboard
Experiments
Results
Discussion
Limitations
Future Work
Conclusion
References
```

---

# Responsibility Handoff

## Completed by the BERT/model team

- Toxicity model
- Sensitive content model
- Hate speech model
- Spam model
- Violence model
- Individual inference scripts
- Unified inference

## Remaining for the other team members

- Feature extraction
- Random Forest severity scorer
- Decision Engine
- Human-in-the-Loop
- Moderator Dashboard
- Full integration
- End-to-end evaluation
- Final documentation/paper

---

# Important

The current system is already capable of running all five BERT classifiers.

The next goal is NOT:

```text
Train another BERT model
```

The next goal is:

```text
Five BERT Outputs
        ↓
Severity Scoring
        ↓
Decision Engine
        ↓
Human Review
        ↓
Moderator Dashboard
```
