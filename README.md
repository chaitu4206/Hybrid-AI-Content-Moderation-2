# Hybrid AI Framework for Automated Content Moderation

## Overview

The Hybrid AI Framework for Automated Content Moderation is a modular AI system for analyzing user-generated content across multiple harmful-content categories.

The current implementation contains five independently trained BERT classifiers:

1. Toxicity Classification
2. Sensitive Content Classification
3. Hate Speech Classification
4. Spam Classification
5. Violence Classification

A unified inference program loads all five trained models and analyzes one input through all classifiers.

## Current Architecture

User Content
    |
    v
Unified Inference Layer
    |
    +--> Toxicity BERT
    +--> Sensitive Content BERT
    +--> Hate Speech BERT
    +--> Spam BERT
    +--> Violence BERT
    |
    v
Combined Classification Outputs
    |
    v
Next Phase: Severity Scoring
    |
    v
Next Phase: Decision Engine
    |
    +--> ALLOW
    +--> WARN
    +--> REMOVE
    +--> ESCALATE
    |
    v
Human-in-the-Loop
    |
    v
Moderator Dashboard

## Completed Components

### Toxicity
Model:
`C:\Users\chait\BERT\models\bert-toxic-model-v3`

The current configuration exposes six output labels:
`LABEL_0` through `LABEL_5`.

The exact semantic mapping should be verified from the training/preprocessing configuration before being named in the final report.

### Sensitive Content
Model:
`C:\Users\chait\models\sensitive_bert`

Classes:
- 0 = NOT SENSITIVE
- 1 = SENSITIVE

Evaluation reported:
- Accuracy: 0.7890
- Macro F1: 0.7886
- Weighted F1: 0.7879

### Hate Speech
Model:
`C:\Users\chait\hatespeech\models\hate_speech_bert`

Classes:
- 0 = HATE SPEECH
- 1 = OFFENSIVE LANGUAGE
- 2 = NEITHER

### Spam
Model:
`C:\Users\chait\spam\models\spam_bert`

Classes:
- 0 = HAM
- 1 = SPAM

### Violence
Model:
`C:\Users\chait\violence\models\violence_bert`

Classes:
- 0 = NON-VIOLENT
- 1 = VIOLENT

The violence dataset used for the experiment was prepared from a public violence-related dataset. The project currently uses a 5,000-sample experimental split:
- Training: 4,000
- Validation: 500
- Testing: 500

## Unified Inference

Project:
`C:\Users\chait\unified_moderation`

Run:

```powershell
cd C:\Users\chait\unified_moderation
python unified_inference.py
```

The program loads all five classifiers and accepts text interactively.

## Model Paths

Toxicity:
`C:\Users\chait\BERT\models\bert-toxic-model-v3`

Sensitive:
`C:\Users\chait\models\sensitive_bert`

Hate Speech:
`C:\Users\chait\hatespeech\models\hate_speech_bert`

Spam:
`C:\Users\chait\spam\models\spam_bert`

Violence:
`C:\Users\chait\violence\models\violence_bert`

## Technologies

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- Pandas
- NumPy
- Scikit-learn
- BERT

## Current Status

| Component | Status |
|---|---|
| Toxicity BERT | Completed |
| Sensitive Content BERT | Completed |
| Hate Speech BERT | Completed |
| Spam BERT | Completed |
| Violence BERT | Completed |
| Unified Inference | Completed |
| Random Forest Severity Scorer | Pending |
| Decision Engine | Pending |
| Human-in-the-Loop | Pending |
| Moderator Dashboard | Pending |
| End-to-End Evaluation | Pending |
| Final Paper/Report | Pending |

## Next Work for Other Members

### 1. Random Forest Severity Scorer

Use outputs from the five BERT classifiers as features.

Potential features:
- Toxicity probabilities
- Sensitive-content probability
- Hate-speech probability
- Offensive-language probability
- Spam probability
- Violence probability

The Random Forest should produce a documented severity level such as:
- LOW
- MEDIUM
- HIGH

The exact labeling and thresholds should be determined from validation data and documented.

### 2. Decision Engine

Combine:
- Five BERT outputs
- Severity score
- Explicit moderation rules

Possible actions:
- ALLOW
- WARN
- REMOVE
- ESCALATE

### 3. Human-in-the-Loop

Ambiguous or high-risk content should be sent to a moderator.

The reviewer should be able to record a final decision.

### 4. Moderator Dashboard

A Streamlit dashboard can display:
- Original content
- Five model predictions
- Probabilities/confidence
- Severity
- Recommended action
- Human review decision

### 5. End-to-End Evaluation

Evaluate:
- Individual model metrics
- Severity scorer metrics
- Decision-engine behavior
- False positives
- False negatives
- Escalation rate

## Repository Guidance

Do not commit large model weights or datasets directly to a normal Git repository.

Suggested `.gitignore`:

```gitignore
__pycache__/
*.pyc
.ipynb_checkpoints/
.venv/
venv/
env/
*.safetensors
*.bin
*.pt
*.pth
checkpoint-*/
*.csv
results/
```

## Project Conclusion

The AI classification foundation is implemented: five specialized BERT classifiers are available and have been successfully connected through a unified inference layer.

The next development stage is to convert the classifier outputs into an end-to-end moderation workflow using severity scoring, decision logic, human review, and a moderator dashboard.
