"""
Hybrid AI Content Moderation
Decision Engine

Combines predictions from:
1. Toxicity
2. Sensitive Content
3. Hate Speech
4. Spam
5. Violence

into a final moderation decision.
"""


def calculate_risk(predictions):
    """
    Calculate overall risk from the five classifiers.

    Parameters
    ----------
    predictions : dict
        Dictionary containing classifier results.

    Returns
    -------
    dict
        Risk score, risk level and moderation action.
    """

    scores = []

    # ---------------------------------------------------------
    # TOXICITY
    # ---------------------------------------------------------

    toxicity = predictions.get("toxicity", {})

    if toxicity:
        toxicity_values = [
            value
            for key, value in toxicity.items()
            if isinstance(value, (int, float))
        ]

        if toxicity_values:
            scores.append(max(toxicity_values))


    # ---------------------------------------------------------
    # SENSITIVE CONTENT
    # ---------------------------------------------------------

    sensitive = predictions.get("sensitive", {})

    if sensitive.get("prediction") == "SENSITIVE":
        scores.append(
            sensitive.get("confidence", 0.0)
        )


    # ---------------------------------------------------------
    # HATE SPEECH
    # ---------------------------------------------------------

    hate = predictions.get("hate_speech", {})

    if hate.get("prediction") == "HATE SPEECH":
        scores.append(
            hate.get("confidence", 0.0)
        )


    # ---------------------------------------------------------
    # SPAM
    # ---------------------------------------------------------

    spam = predictions.get("spam", {})

    if spam.get("prediction") == "SPAM":
        scores.append(
            spam.get("confidence", 0.0)
        )


    # ---------------------------------------------------------
    # VIOLENCE
    # ---------------------------------------------------------

    violence = predictions.get("violence", {})

    if violence.get("prediction") == "VIOLENT":
        scores.append(
            violence.get("confidence", 0.0)
        )


    # ---------------------------------------------------------
    # OVERALL RISK
    # ---------------------------------------------------------

    if not scores:
        risk_score = 0.0
    else:
        risk_score = max(scores)


    # ---------------------------------------------------------
    # RISK LEVEL
    # ---------------------------------------------------------

    if risk_score >= 0.80:

        risk_level = "HIGH"

    elif risk_score >= 0.50:

        risk_level = "MEDIUM"

    else:

        risk_level = "LOW"


    # ---------------------------------------------------------
    # MODERATION ACTION
    # ---------------------------------------------------------

    if risk_level == "LOW":

        action = "ALLOW"

    elif risk_level == "MEDIUM":

        action = "WARN"

    else:

        action = "ESCALATE"


    return {
        "risk_score": round(risk_score, 4),
        "risk_level": risk_level,
        "action": action
    }


# =============================================================
# TEST
# =============================================================

if __name__ == "__main__":

    example_predictions = {

        "toxicity": {
            "LABEL_0": 0.10,
            "LABEL_1": 0.05
        },

        "sensitive": {
            "prediction": "SENSITIVE",
            "confidence": 0.75
        },

        "hate_speech": {
            "prediction": "NEITHER",
            "confidence": 0.90
        },

        "spam": {
            "prediction": "HAM",
            "confidence": 0.95
        },

        "violence": {
            "prediction": "VIOLENT",
            "confidence": 0.65
        }
    }

    result = calculate_risk(
        example_predictions
    )

    print("\nDecision Engine Test")
    print("=" * 50)

    print(
        f"Risk Score : {result['risk_score']:.4f}"
    )

    print(
        f"Risk Level : {result['risk_level']}"
    )

    print(
        f"Action     : {result['action']}"
    )