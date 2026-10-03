# Detailed Project Report
## Hybrid AI Framework for Automated Content Moderation

## 1. Project Overview

The project develops a Hybrid AI Framework for Automated Content Moderation. The goal is to analyze user-generated content and identify multiple forms of potentially harmful content using specialized NLP classifiers.

Instead of depending on a single classifier, the current implementation uses five independently trained BERT models. Each model specializes in a particular classification task.

The five tasks are:
1. Toxicity
2. Sensitive Content
3. Hate Speech
4. Spam
5. Violence

The models are then exposed through a unified inference program.

---

## 2. Problem Statement

User-generated platforms receive large quantities of text content. A moderation system must distinguish between normal content and different types of potentially harmful or inappropriate content.

A single binary classifier is insufficient for the project's intended architecture because content may simultaneously exhibit different characteristics. Therefore, the project uses specialized classifiers and plans to combine their outputs through a severity scorer and decision engine.

---

## 3. Objectives

The current and planned objectives are:

1. Build specialized BERT classifiers for multiple moderation categories.
2. Prepare and preprocess suitable datasets.
3. Train and evaluate the individual models.
4. Store the trained models for inference.
5. Combine the classifiers into one unified inference layer.
6. Build a severity-scoring component.
7. Build a decision engine.
8. Introduce human review for ambiguous/high-risk cases.
9. Build a moderator dashboard.
10. Evaluate the complete moderation pipeline.

---

## 4. System Architecture

The intended final architecture is:

User Content
      |
      v
Unified Inference
      |
      +---- Toxicity BERT
      +---- Sensitive Content BERT
      +---- Hate Speech BERT
      +---- Spam BERT
      +---- Violence BERT
      |
      v
Feature Combination
      |
      v
Random Forest Severity Scorer
      |
      v
Decision Engine
      |
      +---- Allow
      +---- Warn
      +---- Remove
      +---- Escalate
      |
      v
Human Review
      |
      v
Moderator Dashboard

The current implementation reaches the unified inference stage.

---

## 5. Toxicity Classification

The toxicity classifier is stored at:

`C:\Users\chait\BERT\models\bert-toxic-model-v3`

The model is a multi-label BERT classifier.

The current model configuration exposes six labels:
- LABEL_0
- LABEL_1
- LABEL_2
- LABEL_3
- LABEL_4
- LABEL_5

The exact semantic mapping must be verified from the project's training/preprocessing configuration before being stated as named toxicity categories.

The project also contains validation-optimized thresholds under:

`C:\Users\chait\BERT\models\evaluation-v3\best_thresholds.npy`

The inference code uses sigmoid probabilities because the classifier is multi-label.

---

## 6. Sensitive Content Classification

Model:

`C:\Users\chait\models\sensitive_bert`

Classes:
- NOT SENSITIVE
- SENSITIVE

The X-Sensitive dataset was used for training/evaluation.

The recorded test evaluation used 2,000 samples.

Results:

| Metric/Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| Not Sensitive | 0.7180 | 0.8988 | 0.7983 |
| Sensitive | 0.8877 | 0.6937 | 0.7788 |
| Macro Average | 0.8028 | 0.7963 | 0.7886 |
| Weighted Average | 0.8089 | 0.7890 | 0.7879 |

Accuracy:
`0.7890`

The model was also tested through interactive inference.

---

## 7. Hate Speech Classification

Model:

`C:\Users\chait\hatespeech\models\hate_speech_bert`

The model uses three classes:

- HATE SPEECH
- OFFENSIVE LANGUAGE
- NEITHER

The inference program uses softmax probabilities and reports the predicted class, confidence, and class probabilities.

---

## 8. Spam Classification

Model:

`C:\Users\chait\spam\models\spam_bert`

Classes:

- HAM
- SPAM

The inference system returns the predicted class, confidence, and probabilities.

The underlying SMS spam experiment used the SMS Spam Collection dataset.

---

## 9. Violence Classification

Model:

`C:\Users\chait\violence\models\violence_bert`

Classes:

- NON-VIOLENT
- VIOLENT

The project downloaded a public violence-related dataset containing 6,046 records:
- 3,202 non-violent
- 2,844 violent

For the experiment, the preprocessing pipeline subsequently used 5,000 samples:
- 4,000 training
- 500 validation
- 500 testing

Dataset path:

`C:\Users\chait\violence\data\violence\violence_data.csv`

---

## 10. Dataset Preparation

The project uses separate datasets for separate classification tasks.

Each dataset is processed according to its task.

For the violence experiment, the expected CSV structure is:

```csv
text,label
"Peaceful discussion about education.",0
"The attacker violently assaulted the victim.",1
```

The actual experiment should use the public dataset rather than a tiny manually written dataset.

---

## 11. Model Training

The general training process for each classifier is:

1. Load dataset.
2. Inspect columns and class distribution.
3. Clean/prepare text.
4. Split into training/validation/test data where required.
5. Load a BERT tokenizer.
6. Tokenize text.
7. Load a BERT sequence-classification model.
8. Train using the training set.
9. Validate the model.
10. Save the trained model and tokenizer.
11. Evaluate on the test set.
12. Create an inference script.

This process was completed independently for the five classifier types.

---

## 12. Model Storage

The trained models are distributed across the user's local project directories.

### Toxicity
`C:\Users\chait\BERT\models\bert-toxic-model-v3`

### Sensitive
`C:\Users\chait\models\sensitive_bert`

### Hate Speech
`C:\Users\chait\hatespeech\models\hate_speech_bert`

### Spam
`C:\Users\chait\spam\models\spam_bert`

### Violence
`C:\Users\chait\violence\models\violence_bert`

---

## 13. Unified Inference

The unified project is:

`C:\Users\chait\unified_moderation`

The main file is:

`unified_inference.py`

It loads all five models.

The system was successfully tested with an input containing multiple potentially problematic characteristics.

The output included:
- Toxicity probabilities
- Sensitive-content prediction
- Hate-speech prediction
- Spam prediction
- Violence prediction

This proves that the five independent classifiers can operate behind one common inference interface.

---

## 14. Important Architectural Point

The five BERT models are not physically merged into one neural network.

They remain independent specialist models.

The unified system is an inference orchestration layer:

Input
 -> Toxicity model
 -> Sensitive model
 -> Hate-speech model
 -> Spam model
 -> Violence model
 -> Combined outputs

This modular design allows each classifier to be trained, evaluated, replaced, or improved independently.

---

## 15. Current Limitations

The current implementation does not yet provide a complete moderation decision.

It currently reports classifier predictions.

It does not yet contain:
- Random Forest severity scoring
- Final decision rules
- Human moderator workflow
- Moderator dashboard
- Complete end-to-end evaluation

These are the next development tasks.

---

# PART B — WORK TO BE DONE BY OTHER MEMBERS

## 16. Random Forest Severity Scorer

The next component should take outputs from the five BERT classifiers.

Potential features:

```text
toxicity scores
sensitive probability
hate speech probability
offensive language probability
spam probability
violence probability
```

The features can be represented as a numerical vector.

Example:

```text
[
  toxicity_score,
  sensitive_score,
  hate_score,
  offensive_score,
  spam_score,
  violence_score
]
```

The Random Forest should classify overall severity.

Possible severity classes:
- LOW
- MEDIUM
- HIGH

The final team implementation should document how severity labels are assigned and validated.

---

## 17. Decision Engine

The Decision Engine should transform model/severity information into a moderation action.

Conceptually:

Model Outputs
     |
     v
Severity Score
     |
     v
Decision Rules
     |
     +--> ALLOW
     +--> WARN
     +--> REMOVE
     +--> ESCALATE

The rules should be explicit and documented.

---

## 18. Human-in-the-Loop

Automated moderation should allow human review for cases that are uncertain or require contextual judgment.

A reviewer interface should show:
- Original text
- Model predictions
- Confidence/probability
- Severity
- Suggested action

The reviewer should then record a final decision.

---

## 19. Moderator Dashboard

A Streamlit dashboard can provide the user interface.

Suggested sections:

### Content
Display the submitted text.

### Classifier Results
Display:
- Toxicity
- Sensitive content
- Hate speech
- Spam
- Violence

### Severity
Display:
- Low
- Medium
- High

### Decision
Display:
- Allow
- Warn
- Remove
- Escalate

### Human Review
Provide controls for a moderator to confirm or override the automated recommendation.

---

## 20. Final Pipeline Integration

After the Random Forest, Decision Engine, and dashboard are implemented, the complete pipeline should become:

Input
 |
 v
Five BERT Models
 |
 v
Combined Features
 |
 v
Random Forest Severity
 |
 v
Decision Engine
 |
 +--> Allow
 +--> Warn
 +--> Remove
 +--> Escalate
          |
          v
    Human Review
          |
          v
      Final Action

---

## 21. Final Evaluation

The final project should evaluate both individual components and the complete system.

Individual classifier metrics:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

Severity scorer:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

End-to-end moderation:
- Correct decisions
- False positives
- False negatives
- Escalation rate
- Human override rate

---

## 22. Research Paper / Report Structure

The final report can use:

1. Abstract
2. Introduction
3. Problem Statement
4. Objectives
5. Literature Review
6. Dataset Description
7. Data Preprocessing
8. Methodology
9. Toxicity Classifier
10. Sensitive Content Classifier
11. Hate Speech Classifier
12. Spam Classifier
13. Violence Classifier
14. Unified Inference
15. Severity Scoring
16. Decision Engine
17. Human-in-the-Loop
18. Moderator Dashboard
19. Experimental Results
20. Discussion
21. Limitations
22. Future Work
23. Conclusion
24. References

---

## 23. Current Project Status

| Component | Status |
|---|---|
| Toxicity BERT | Completed |
| Sensitive BERT | Completed |
| Hate Speech BERT | Completed |
| Spam BERT | Completed |
| Violence BERT | Completed |
| Unified Inference | Completed |
| Random Forest | Pending |
| Decision Engine | Pending |
| Human-in-the-Loop | Pending |
| Moderator Dashboard | Pending |
| Full Pipeline | Pending |
| Final Evaluation | Pending |
| Final Report/Paper | Pending |

---

## 24. Conclusion

The current phase has established the core AI classification layer of the Hybrid AI Content Moderation Framework.

Five specialized BERT models have been trained and stored separately. A unified inference program has successfully loaded and executed all five models against a single input.

The remaining work is primarily system integration: severity scoring, decision logic, human review, dashboard development, and end-to-end evaluation.

These components will convert the current multi-model classifier into the complete hybrid moderation framework proposed by the project.
