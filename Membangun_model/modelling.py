import pandas as pd
import numpy as np
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
import os
import joblib

warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('data_clean.csv')
X = df.drop(columns=['difficulty_rank_encoded'])
y = df['difficulty_rank_encoded']

# Split data — PASTIKAN INI SAMA untuk prediksi dan CM
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

# Setup MLflow
mlflow.set_experiment("Japanese_Universities_Skilled")

with mlflow.start_run(run_name="Tuned_RandomForest") as run:
    
    # GridSearchCV with threading backend
    with joblib.parallel_backend('threading', n_jobs=2):
        base_model = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(
            base_model, param_grid, cv=3, scoring='accuracy'
        )
        grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    # Log best params
    for param, value in best_params.items():
        mlflow.log_param(param, value)
    mlflow.log_metric("best_cv_score", grid_search.best_score_)
    
    # ============================================
    # PREDIKSI — PAKAI YANG SAMA UNTUK SEMUA METRIK
    # ============================================
    y_pred = best_model.predict(X_test)           # ← Prediksi class
    y_pred_proba = best_model.predict_proba(X_test)  # ← Prediksi probability
    
    # Hitung metrik — semua dari y_test dan y_pred yang SAMA
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
    
    # Log model
    mlflow.sklearn.log_model(best_model, "random_forest_model")
    
    # ============================================
    # CONFUSION MATRIX — PAKAI y_test DAN y_pred YANG SAMA!
    # ============================================
    
    # 1. RAW confusion matrix (counts)
    cm_raw = confusion_matrix(y_test, y_pred)
    print("\n=== RAW Confusion Matrix ===")
    print(cm_raw)
    print(f"Trace (correct predictions): {np.trace(cm_raw)}")
    print(f"Total: {cm_raw.sum()}")
    print(f"Accuracy from CM: {np.trace(cm_raw) / cm_raw.sum():.4f}")
    
    # 2. NORMALIZED confusion matrix (for visualization)
    cm_normalized = confusion_matrix(y_test, y_pred, normalize='true')
    
    # Plot normalized CM
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm_normalized, 
        annot=True, 
        fmt='.2f',  # 2 decimal places untuk normalized
        cmap='Blues',
        vmin=0, 
        vmax=1
    )
    plt.title("Normalized Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.close()
    mlflow.log_artifact("confusion_matrix.png")
    os.remove("confusion_matrix.png")
    
    # ============================================
    # FEATURE IMPORTANCE
    # ============================================
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
    
    # Debug: classification report
    print(f"\n=== Classification Report ===")
    print(classification_report(y_test, y_pred))
    
    print(f"\nBest params: {best_params}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")
    print(f"Test Accuracy: {acc:.4f}")
    run_id = run.info.run_id

print("Modelling tuning selesai. Run ID:", run_id)