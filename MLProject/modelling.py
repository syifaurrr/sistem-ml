import argparse
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import mlflow
import mlflow.sklearn
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data_clean.csv")
    parser.add_argument("--n_estimators", type=int, default=200)
    parser.add_argument("--max_depth", type=int, default=20)
    parser.add_argument("--min_samples_split", type=int, default=10)
    return parser.parse_args()


def find_data_file(filename):
    if os.path.exists(filename):
        return filename
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for dir_path in [script_dir, os.path.dirname(script_dir), os.getcwd()]:
        for name in [filename, os.path.basename(filename)]:
            full_path = os.path.join(dir_path, name)
            if os.path.exists(full_path):
                return full_path
    raise FileNotFoundError(f"Data file '{filename}' not found.")


def main():
    args = parse_args()
    data_path = find_data_file(args.data_path)
    print(f"Using data file: {data_path}")
    
    df = pd.read_csv(data_path)
    X = df.drop(columns=['difficulty_rank_encoded'])
    y = df['difficulty_rank_encoded']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    param_grid = {
        'n_estimators': [args.n_estimators],
        'max_depth': [args.max_depth if args.max_depth > 0 else None],
        'min_samples_split': [args.min_samples_split]
    }

    # === FIX: Jangan set_experiment dan jangan start_run! ===
    # mlflow run sudah handle semua ini
    # Langsung training dan log saja

    with joblib.parallel_backend('threading', n_jobs=2):
        base_model = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(base_model, param_grid, cv=3, scoring='accuracy')
        grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    # Log langsung ke run yang sudah aktif (dari mlflow run)
    for param, value in grid_search.best_params_.items():
        mlflow.log_param(param, value)
    mlflow.log_metric("best_cv_score", grid_search.best_score_)

    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    try:
        roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
        mlflow.log_metric("roc_auc_ovr", roc_auc)
    except ValueError:
        pass

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision_weighted", prec)
    mlflow.log_metric("recall_weighted", rec)
    mlflow.log_metric("f1_weighted", f1)

    mlflow.sklearn.log_model(best_model, "random_forest_model")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.close()
    mlflow.log_artifact("confusion_matrix.png")
    os.remove("confusion_matrix.png")

    # Feature Importance
    importances = best_model.feature_importances_
    features = X.columns
    indices = np.argsort(importances)[::-1][:10]

    plt.figure(figsize=(10, 6))
    plt.barh(range(10), importances[indices][::-1], align='center')
    plt.yticks(range(10), [features[i] for i in indices[::-1]])
    plt.xlabel("Importance")
    plt.title("Top 10 Feature Importances")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150)
    plt.close()
    mlflow.log_artifact("feature_importance.png")
    os.remove("feature_importance.png")

    # Get run info dari run yang sudah aktif
    run = mlflow.active_run()
    if run:
        run_id = run.info.run_id
        with open("run_id.txt", "w") as f:
            f.write(run_id)
        mlflow.log_artifact("run_id.txt")
        os.remove("run_id.txt")
        print(f"\nRun ID: {run_id}")

    print(f"Best params: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")
    print(f"Test Accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()