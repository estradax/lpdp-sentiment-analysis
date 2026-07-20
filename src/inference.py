import os
import sys
import logging
from typing import Dict, Any, Union, List
import joblib
import pandas as pd
import numpy as np

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Add the project root directory to python path if executing as a script directly
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing import TextPreprocessor
from src.lexicon import LexiconLabeler

logger = logging.getLogger(__name__)


class SentimentPredictor:
    """Predicts sentiment category of raw Indonesian texts using a trained model."""
    
    def __init__(self, model_dir: str = "saved_model"):
        """Loads serialized model files and initializes text preprocessor."""
        self.model_dir = model_dir
        
        best_model_path = os.path.join(model_dir, "best_model.pkl")
        tfidf_path = os.path.join(model_dir, "tfidf_vectorizer.pkl")
        le_path = os.path.join(model_dir, "label_encoder.pkl")
        
        if not (os.path.exists(best_model_path) and os.path.exists(tfidf_path) and os.path.exists(le_path)):
            raise FileNotFoundError(
                f"Model artifacts not found in {model_dir}. Please run training pipeline first."
            )
            
        logger.info(f"Loading model artifacts from {model_dir}...")
        self.model = joblib.load(best_model_path)
        self.tfidf = joblib.load(tfidf_path)
        self.label_encoder = joblib.load(le_path)
        
        # Initialize text preprocessor and lexicon labeler
        self.preprocessor = TextPreprocessor()
        self.labeler = LexiconLabeler()
        logger.info("SentimentPredictor is ready.")

    def predict(self, text: str) -> Dict[str, Any]:
        """Predicts sentiment for a single input string.
        
        Process flow:
        Raw Text -> Clean -> Case Fold -> Normalize -> Remove Stopwords -> Stem -> TF-IDF + Lexicon Blending -> Predict
        
        Note: Language filtering is disabled during inference to prevent dropping
        short queries.
        """
        if not isinstance(text, str) or not text.strip():
            return self._empty_result(text)
            
        # Clean and preprocess text (disabling language check for inference safety)
        clean_text = self.preprocessor.filter_text(text)
        lower_text = clean_text.lower()
        normalized = self.preprocessor.normalize_words(lower_text)
        no_stop = self.preprocessor.remove_stopwords(normalized)
        stemmed = self.preprocessor.stem(no_stop)
        
        # Handle case where text becomes empty after cleaning
        if not stemmed.strip():
            return self._empty_result(text)
            
        # 1. Calculate lexicon score and pseudo-probabilities
        lex_score, _ = self.labeler.label_sentiment(stemmed)
        classes = self.label_encoder.classes_
        class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        
        lex_proba = np.zeros(len(classes))
        if lex_score > 0:
            lex_proba[class_to_idx.get("Positif", 0)] = 0.85 if lex_score < 5 else 0.95
            lex_proba[class_to_idx.get("Netral", 0)] = 0.10 if lex_score < 5 else 0.04
            lex_proba[class_to_idx.get("Negatif", 0)] = 0.05 if lex_score < 5 else 0.01
        elif lex_score < 0:
            lex_proba[class_to_idx.get("Negatif", 0)] = 0.85 if lex_score > -5 else 0.95
            lex_proba[class_to_idx.get("Netral", 0)] = 0.10 if lex_score > -5 else 0.04
            lex_proba[class_to_idx.get("Positif", 0)] = 0.05 if lex_score > -5 else 0.01
        else:
            lex_proba[class_to_idx.get("Netral", 0)] = 0.80
            lex_proba[class_to_idx.get("Positif", 0)] = 0.10
            lex_proba[class_to_idx.get("Negatif", 0)] = 0.10
            
        # 2. Vectorize using loaded TF-IDF and get model probabilities
        features = self.tfidf.transform([stemmed])
        model_proba = self.model.predict_proba(features)[0]
        
        # 3. Blend model and lexicon probabilities
        final_proba = 0.5 * model_proba + 0.5 * lex_proba
        pred_idx = np.argmax(final_proba)
        pred_label = classes[pred_idx]
        
        prob_detail = {
            str(cls): round(float(p), 4)
            for cls, p in zip(classes, final_proba)
        }
        
        return {
            "text_original": text,
            "text_preprocessed": stemmed,
            "sentiment": str(pred_label),
            "confidence": round(float(final_proba.max()), 4),
            "probabilities": prob_detail,
        }

    def predict_batch(self, texts: List[str]) -> pd.DataFrame:
        """Batch predicts a list of texts and returns a Pandas DataFrame."""
        logger.info(f"Batch predicting {len(texts)} texts...")
        
        classes = self.label_encoder.classes_
        class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        
        # 1. Preprocess all texts
        preprocessed_texts = []
        valid_indices = []
        results = [None] * len(texts)
        
        for idx, text in enumerate(texts):
            if not isinstance(text, str) or not text.strip():
                results[idx] = self._empty_result(text)
                continue
                
            clean_text = self.preprocessor.filter_text(text)
            lower_text = clean_text.lower()
            normalized = self.preprocessor.normalize_words(lower_text)
            no_stop = self.preprocessor.remove_stopwords(normalized)
            stemmed = self.preprocessor.stem(no_stop)
            
            if not stemmed.strip():
                results[idx] = self._empty_result(text)
            else:
                preprocessed_texts.append(stemmed)
                valid_indices.append(idx)
                
        # 2. Vectorize and predict in batch if there are valid texts
        if preprocessed_texts:
            features = self.tfidf.transform(preprocessed_texts)
            probas = self.model.predict_proba(features)
            
            for i, idx in enumerate(valid_indices):
                stemmed = preprocessed_texts[i]
                lex_score, _ = self.labeler.label_sentiment(stemmed)
                
                # Determine lexicon pseudo-probabilities
                lex_proba = np.zeros(len(classes))
                if lex_score > 0:
                    lex_proba[class_to_idx.get("Positif", 0)] = 0.85 if lex_score < 5 else 0.95
                    lex_proba[class_to_idx.get("Netral", 0)] = 0.10 if lex_score < 5 else 0.04
                    lex_proba[class_to_idx.get("Negatif", 0)] = 0.05 if lex_score < 5 else 0.01
                elif lex_score < 0:
                    lex_proba[class_to_idx.get("Negatif", 0)] = 0.85 if lex_score > -5 else 0.95
                    lex_proba[class_to_idx.get("Netral", 0)] = 0.10 if lex_score > -5 else 0.04
                    lex_proba[class_to_idx.get("Positif", 0)] = 0.05 if lex_score > -5 else 0.01
                else:
                    lex_proba[class_to_idx.get("Netral", 0)] = 0.80
                    lex_proba[class_to_idx.get("Positif", 0)] = 0.10
                    lex_proba[class_to_idx.get("Negatif", 0)] = 0.10
                    
                model_proba = probas[i]
                final_proba = 0.5 * model_proba + 0.5 * lex_proba
                pred_idx = np.argmax(final_proba)
                pred_label = classes[pred_idx]
                
                prob_detail = {
                    str(cls): round(float(p), 4)
                    for cls, p in zip(classes, final_proba)
                }
                
                results[idx] = {
                    "text_original": texts[idx],
                    "text_preprocessed": stemmed,
                    "sentiment": str(pred_label),
                    "confidence": round(float(final_proba.max()), 4),
                    "probabilities": prob_detail,
                }
                
        # 3. Flatten probabilities for clean DataFrame columns
        flat_results = []
        for res in results:
            if res is None:
                continue
            row = {
                "text_original": res["text_original"],
                "text_preprocessed": res["text_preprocessed"],
                "sentiment": res["sentiment"],
                "confidence": res["confidence"],
            }
            for cls, prob in res["probabilities"].items():
                row[f"prob_{cls}"] = prob
            flat_results.append(row)
            
        return pd.DataFrame(flat_results)

    def _empty_result(self, original_text: str) -> Dict[str, Any]:
        """Utility function for mapping empty or cleaned-out text."""
        classes = self.label_encoder.classes_
        return {
            "text_original": original_text,
            "text_preprocessed": "",
            "sentiment": "Netral",
            "confidence": 0.0,
            "probabilities": {str(cls): 0.0 for cls in classes},
        }


class BertSentimentPredictor:
    """Predicts sentiment category of raw Indonesian texts using a fine-tuned IndoBERT model."""
    
    def __init__(self, model_dir: str = "saved_model_indobert", max_length: int = 128):
        """Loads fine-tuned model and tokenizer."""
        self.model_dir = model_dir
        self.max_length = max_length
        
        best_model_path = os.path.join(model_dir, "model.safetensors")
        # Check pytorch model bin just in case it was saved in old format
        pytorch_model_path = os.path.join(model_dir, "pytorch_model.bin")
        le_path = os.path.join(model_dir, "label_encoder.pkl")
        
        if not (os.path.exists(best_model_path) or os.path.exists(pytorch_model_path)) or not os.path.exists(le_path):
            raise FileNotFoundError(
                f"BERT Model artifacts not found in {model_dir}. Please run fine-tuning pipeline first."
            )
            
        logger.info(f"Loading BERT model and tokenizer from {model_dir}...")
        self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        logger.info(f"Inference device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()
        
        self.label_encoder = joblib.load(le_path)
        self.preprocessor = TextPreprocessor()
        logger.info("BertSentimentPredictor is ready.")

    def predict(self, text: str) -> Dict[str, Any]:
        """Predicts sentiment for a single input string using BERT."""
        if not isinstance(text, str) or not text.strip():
            return self._empty_result(text)
            
        # Clean and preprocess text lightweight (no stopword removal / stemming)
        clean_text = self.preprocessor.preprocess_for_bert(text, filter_lang=False)
        
        if not clean_text.strip():
            return self._empty_result(text)
            
        inputs = self.tokenizer(
            clean_text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.max_length,
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
            
        classes = self.label_encoder.classes_
        pred_idx = int(np.argmax(probs))
        pred_label = classes[pred_idx]
        
        prob_detail = {
            str(cls): round(float(p), 4)
            for cls, p in zip(classes, probs)
        }
        
        return {
            "text_original": text,
            "text_preprocessed": clean_text,
            "sentiment": str(pred_label),
            "confidence": round(float(probs.max()), 4),
            "probabilities": prob_detail,
        }

    def predict_batch(self, texts: List[str]) -> pd.DataFrame:
        """Batch predicts a list of texts using BERT and returns a DataFrame."""
        logger.info(f"Batch predicting {len(texts)} texts using BERT...")
        results = []
        for text in texts:
            results.append(self.predict(text))
            
        flat_results = []
        for res in results:
            row = {
                "text_original": res["text_original"],
                "text_preprocessed": res["text_preprocessed"],
                "sentiment": res["sentiment"],
                "confidence": res["confidence"],
            }
            for cls, prob in res["probabilities"].items():
                row[f"prob_{cls}"] = prob
            flat_results.append(row)
            
        return pd.DataFrame(flat_results)

    def _empty_result(self, original_text: str) -> Dict[str, Any]:
        """Utility function for mapping empty or cleaned-out text."""
        classes = self.label_encoder.classes_
        return {
            "text_original": original_text,
            "text_preprocessed": "",
            "sentiment": "Netral",
            "confidence": 0.0,
            "probabilities": {str(cls): 0.0 for cls in classes},
        }


def main():
    """CLI handler for running prediction."""
    import argparse
    import json
    
    # Simple console logger
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    
    parser = argparse.ArgumentParser(description="LPDP Twitter Sentiment Inference Engine")
    parser.add_argument(
        "--text",
        type=str,
        help="Single text snippet to predict sentiment for"
    )
    parser.add_argument(
        "--input-file",
        type=str,
        help="Path to CSV or TXT file containing texts for batch prediction"
    )
    parser.add_argument(
        "--column",
        type=str,
        default="text",
        help="Column name in CSV file to predict on (default: 'text')"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        help="Path to save batch prediction results (CSV format)"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="saved_model",
        help="Directory where model artifacts are saved (default: 'saved_model')"
    )
    
    args = parser.parse_args()
    
    try:
        predictor = SentimentPredictor(model_dir=args.model_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
        
    if args.text:
        result = predictor.predict(args.text)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
    elif args.input_file:
        if not os.path.exists(args.input_file):
            print(f"Error: Input file {args.input_file} not found.")
            return
            
        # Determine format
        if args.input_file.endswith(".csv"):
            df = pd.read_csv(args.input_file)
            if args.column not in df.columns:
                print(f"Error: Column '{args.column}' not found in CSV. Available columns: {list(df.columns)}")
                return
            texts = df[args.column].astype(str).tolist()
        else:
            # Assume text file where each line is a sentence
            with open(args.input_file, "r", encoding="utf-8") as f:
                texts = [line.strip() for line in f if line.strip()]
                
        results_df = predictor.predict_batch(texts)
        
        if args.output_file:
            results_df.to_csv(args.output_file, index=False)
            print(f"Batch prediction results saved to {args.output_file}")
        else:
            print(results_df.to_string(index=False))
            
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
