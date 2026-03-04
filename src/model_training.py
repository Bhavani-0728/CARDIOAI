import joblib
import json
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score
)


def train_model(df):

    df = df.drop("id", axis=1)

    X = df.drop("cardio", axis=1)
    y = df["cardio"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Faster RF
    model = RandomForestClassifier(
        n_estimators=150,        # reduced
        max_depth=15,            # limit depth
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    print("Accuracy:", accuracy_score(y_test, preds))
    print("\nClassification Report:\n")
    print(classification_report(y_test, preds))
    print("ROC-AUC:", roc_auc_score(y_test, probs))

    joblib.dump(model, "models/best_model.pkl")

    with open("models/feature_columns.json", "w") as f:
        json.dump(list(X.columns), f)

    importances = model.feature_importances_

    plt.figure(figsize=(10, 6))
    plt.barh(X.columns, importances)
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.savefig("models/feature_importance.png")
    plt.close()

    print("\nFeature importance saved.")