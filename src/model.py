import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

logger = logging.getLogger(__name__)


def build_tfidf_features(
    texts: pd.Series,
    max_features: int = 20000,
    min_df: int = 3,
    ngram_range: Tuple[int, int] = (1, 2)
) -> Tuple[Any, Any]:
    """Fits TF-IDF vectorizer and transforms text corpus."""
    logger.info("Initializing and fitting TF-IDF Vectorizer...")
    tfidf = TfidfVectorizer(
        sublinear_tf=True,
        min_df=min_df,
        max_features=max_features,
        ngram_range=ngram_range,
        analyzer="word",
    )
    X_tfidf = tfidf.fit_transform(texts.fillna(""))
    logger.info(f"TF-IDF Matrix Shape: {X_tfidf.shape}, Vocabulary Size: {len(tfidf.vocabulary_)}")
    return tfidf, X_tfidf


def encode_labels(labels: pd.Series) -> Tuple[LabelEncoder, np.ndarray, Dict[str, int]]:
    """Encodes categorical labels to integer targets."""
    logger.info("Encoding categorical labels...")
    le = LabelEncoder()
    y = le.fit_transform(labels)
    label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    logger.info(f"Encoded classes: {label_mapping}")
    return le, y, label_mapping


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, classes: np.ndarray) -> Dict[str, Any]:
    """Computes standard evaluation metrics."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro")),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro")),
    }


def save_confusion_matrix(
    cm: np.ndarray,
    classes: list,
    title: str,
    output_path: str,
    cmap: str = "Blues"
) -> None:
    """Plots and saves confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap=cmap,
        xticklabels=classes, yticklabels=classes,
        linewidths=0.5, linecolor="white",
        annot_kws={"size": 12, "weight": "bold"},
    )
    ax.set_xlabel("Prediksi", fontsize=11)
    ax.set_ylabel("Aktual", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved confusion matrix to {output_path}")


def save_feature_importances(
    importances: np.ndarray,
    feature_names: np.ndarray,
    title: str,
    output_path: str,
    top_n: int = 20
) -> pd.DataFrame:
    """Plots and saves horizontal bar chart of top N feature importances."""
    top_idx = np.argsort(importances)[::-1][:top_n]
    df_imp = pd.DataFrame({
        "Fitur": feature_names[top_idx],
        "Importance": importances[top_idx],
    })
    
    fig, ax = plt.subplots(figsize=(12, 7))
    palette = sns.color_palette("flare", len(df_imp))
    ax.barh(df_imp["Fitur"][::-1], df_imp["Importance"][::-1], color=palette)
    ax.set_xlabel("Feature Importance Score", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    
    for i, (val, name) in enumerate(zip(df_imp["Importance"][::-1], df_imp["Fitur"][::-1])):
        ax.text(val + 0.0001, i, f"{val:.4f}", va="center", fontsize=8)
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved feature importances to {output_path}")
    return df_imp


def train_pipeline(
    df: pd.DataFrame,
    model_dir: str = "saved_model",
    reports_dir: str = "reports",
    run_tuning: bool = True,
    random_state: int = 42
) -> Tuple[Any, Any, Any, Dict[str, Any]]:
    """Runs the training, evaluation, and saving pipeline.
    
    1. Prepares features (TF-IDF) and targets (Label Encoder).
    2. Stratified train-test split (80/20).
    3. Trains base Random Forest.
    4. (Optional) RandomizedSearch tuning.
    5. Evaluates model and writes performance reports/plots.
    6. Serializes model artifacts.
    """
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    
    # ── Feature Vectorization ──────────────────────────
    tfidf, X_tfidf = build_tfidf_features(df["processed_text"])
    le, y, label_mapping = encode_labels(df["label"])
    
    # ── Train-Test Split ────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X_tfidf, y,
        test_size=0.2,
        random_state=random_state,
        stratify=y,
    )
    
    # ── Base Model Training ─────────────────────────────
    logger.info("Training Base Random Forest Classifier...")
    rf_base = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        n_jobs=-1,
        random_state=random_state,
    )
    rf_base.fit(X_train, y_train)
    
    # Evaluate Base Model
    y_pred_base = rf_base.predict(X_test)
    metrics_base = evaluate_model(y_test, y_pred_base, le.classes_)
    logger.info(f"Base Model Accuracy: {metrics_base['accuracy']:.4%}")
    
    # Save Base Confusion Matrix
    cm_base = confusion_matrix(y_test, y_pred_base)
    save_confusion_matrix(
        cm_base, list(le.classes_),
        f"Confusion Matrix — Base Model (Akurasi: {metrics_base['accuracy']*100:.2f}%)",
        os.path.join(reports_dir, "confusion_matrix_base.png"),
        cmap="Blues"
    )
    
    # Base Classification Report to txt
    base_report_str = classification_report(y_test, y_pred_base, target_names=le.classes_)
    with open(os.path.join(reports_dir, "classification_report_base.txt"), "w") as f:
        f.write(base_report_str)
        
    final_model = rf_base
    metrics_final = metrics_base
    y_pred_final = y_pred_base
    best_params = {
        "n_estimators": rf_base.n_estimators,
        "max_depth": rf_base.max_depth,
        "min_samples_split": rf_base.min_samples_split,
        "min_samples_leaf": rf_base.min_samples_leaf,
        "max_features": rf_base.max_features,
        "bootstrap": rf_base.bootstrap,
    }
    
    # ── Hyperparameter Tuning ──────────────────────────
    if run_tuning:
        logger.info("Running RandomizedSearchCV hyperparameter tuning...")
        param_distributions = {
            "n_estimators": [100, 200, 300, 400, 500],
            "max_depth": [None, 10, 20, 30, 50],
            "min_samples_split": [2, 5, 10, 15],
            "min_samples_leaf": [1, 2, 4, 6],
            "max_features": ["sqrt", "log2", 0.3, 0.5],
            "bootstrap": [True, False],
        }
        rf_tuning = RandomForestClassifier(
            class_weight="balanced_subsample",
            n_jobs=None,
            random_state=random_state,
        )
        search = RandomizedSearchCV(
            estimator=rf_tuning,
            param_distributions=param_distributions,
            n_iter=30,
            scoring="f1_macro",
            cv=5,
            refit=True,
            n_jobs=-1,
            random_state=random_state,
            verbose=1,
        )
        search.fit(X_train, y_train)
        
        final_model = search.best_estimator_
        best_params = search.best_params_
        best_cv_score = search.best_score_
        
        logger.info(f"Tuning finished. Best CV F1-macro Score: {best_cv_score:.4f}")
        logger.info(f"Best hyperparameters: {best_params}")
        
        # Evaluate Tuned Model
        y_pred_final = final_model.predict(X_test)
        metrics_final = evaluate_model(y_test, y_pred_final, le.classes_)
        logger.info(f"Tuned Model Accuracy: {metrics_final['accuracy']:.4%}")
        
        # Save Tuned Confusion Matrix
        cm_tuned = confusion_matrix(y_test, y_pred_final)
        save_confusion_matrix(
            cm_tuned, list(le.classes_),
            f"Confusion Matrix — Tuned Model (Akurasi: {metrics_final['accuracy']*100:.2f}%)",
            os.path.join(reports_dir, "confusion_matrix_tuned.png"),
            cmap="Greens"
        )
        
        # Tuned Classification Report to txt
        tuned_report_str = classification_report(y_test, y_pred_final, target_names=le.classes_)
        with open(os.path.join(reports_dir, "classification_report_tuned.txt"), "w") as f:
            f.write(tuned_report_str)
            
    # ── Feature Importances ─────────────────────────────
    feat_names = tfidf.get_feature_names_out()
    imp_df = save_feature_importances(
        final_model.feature_importances_,
        feat_names,
        "Top-20 Fitur TF-IDF Paling Berpengaruh (Random Forest)",
        os.path.join(reports_dir, "feature_importances.png"),
        top_n=20
    )
    imp_df.to_csv(os.path.join(reports_dir, "feature_importances.csv"), index=False)
    
    # ── Save Model Comparison ───────────────────────────
    comparison_rows = [
        {
            "Model": "Random Forest Base",
            "Akurasi": f"{metrics_base['accuracy']:.4f}",
            "F1 Macro": f"{metrics_base['f1_macro']:.4f}",
            "F1 Weighted": f"{metrics_base['f1_weighted']:.4f}",
            "Precision Macro": f"{metrics_base['precision_macro']:.4f}",
            "Recall Macro": f"{metrics_base['recall_macro']:.4f}",
        }
    ]
    if run_tuning:
        comparison_rows.append({
            "Model": "Random Forest Tuned",
            "Akurasi": f"{metrics_final['accuracy']:.4f}",
            "F1 Macro": f"{metrics_final['f1_macro']:.4f}",
            "F1 Weighted": f"{metrics_final['f1_weighted']:.4f}",
            "Precision Macro": f"{metrics_final['precision_macro']:.4f}",
            "Recall Macro": f"{metrics_final['recall_macro']:.4f}",
        })
    df_compare = pd.DataFrame(comparison_rows)
    df_compare.to_csv(os.path.join(reports_dir, "model_comparison.csv"), index=False)
    
    # Print model comparison to log
    logger.info("\n" + "="*50 + "\nMODEL COMPARISON SUMMARY\n" + "="*50 + "\n" + df_compare.to_string(index=False) + "\n" + "="*50)
    
    # ── Serialize Artifacts ──────────────────────────────
    model_path = os.path.join(model_dir, "best_model.pkl")
    tfidf_path = os.path.join(model_dir, "tfidf_vectorizer.pkl")
    le_path = os.path.join(model_dir, "label_encoder.pkl")
    meta_path = os.path.join(model_dir, "metadata.json")
    
    joblib.dump(final_model, model_path)
    joblib.dump(tfidf, tfidf_path)
    joblib.dump(le, le_path)
    
    # Convert types in best_params to be JSON serializable (None -> "None", etc)
    serializable_params = {}
    for k, v in best_params.items():
        if v is None:
            serializable_params[k] = "None"
        elif isinstance(v, (np.integer, np.floating)):
            serializable_params[k] = v.item()
        else:
            serializable_params[k] = v
            
    metadata = {
        "created_at": datetime.now().isoformat(),
        "model_type": "RandomForestClassifier",
        "classes": le.classes_.tolist(),
        "label_mapping": {k: int(v) for k, v in label_mapping.items()},
        "best_params": serializable_params,
        "performance": {
            "base_model_accuracy": round(metrics_base["accuracy"], 4),
            "final_model_accuracy": round(metrics_final["accuracy"], 4),
            "f1_macro_final": round(metrics_final["f1_macro"], 4),
        },
        "vectorizer_config": {
            "max_features": tfidf.max_features,
            "ngram_range": list(tfidf.ngram_range),
            "sublinear_tf": tfidf.sublinear_tf,
            "min_df": tfidf.min_df,
        },
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
        
    logger.info(f"Successfully saved all model artifacts to {model_dir}/")
    return final_model, tfidf, le, metadata
