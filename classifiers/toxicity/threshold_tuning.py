import numpy as np

from sklearn.metrics import f1_score

from preprocessing import LABELS


# =========================================================
# LOAD RESULTS
# =========================================================

data = np.load(
    "../models/evaluation/test_predictions.npz"
)

probabilities = data["probabilities"]

labels = data["labels"]


# =========================================================
# FIND BEST THRESHOLD
# =========================================================

print("\n========================================")
print("THRESHOLD OPTIMIZATION")
print("========================================")


best_thresholds = {}


for i, label in enumerate(LABELS):

    best_threshold = 0.5
    best_f1 = 0.0

    for threshold in np.arange(
        0.10,
        0.91,
        0.05
    ):

        predictions = (
            probabilities[:, i]
            >= threshold
        ).astype(int)

        score = f1_score(
            labels[:, i],
            predictions,
            zero_division=0
        )

        if score > best_f1:

            best_f1 = score

            best_threshold = threshold


    best_thresholds[label] = best_threshold

    print(
        f"{label:<15} "
        f"Threshold: {best_threshold:.2f} "
        f"F1: {best_f1:.4f}"
    )


# =========================================================
# SAVE THRESHOLDS
# =========================================================

np.save(
    "../models/evaluation/best_thresholds.npy",
    best_thresholds
)

print(
    "\nThresholds saved successfully."
)