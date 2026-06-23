import os
import sys
import logging
import argparse
import pandas as pd
from datetime import datetime

# Configure standard logging to console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("lpdp_rf_pipeline")

from src.preprocessing import TextPreprocessor
from src.lexicon import LexiconLabeler
from src.eda import run_all_eda
from src.model import train_pipeline
from src.inference import SentimentPredictor


def run_pipeline(
    dataset_url: str,
    skip_tuning: bool,
    model_dir: str,
    reports_dir: str,
    run_test: bool
) -> None:
    """Orchestrates the entire sentiment analysis and modeling process."""
    start_time = datetime.now()
    logger.info("=" * 60)
    logger.info("🚀 STARTING LPDP SENTIMENT ANALYSIS PIPELINE")
    logger.info("=" * 60)
    
    # ── Step 1: Fetch and Load Dataset ───────────────────────────
    logger.info(f"Step 1: Loading dataset from GitHub URL: {dataset_url} ...")
    try:
        df = pd.read_csv(dataset_url, low_memory=False)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        sys.exit(1)
        
    required_cols = ["id_str", "created_at", "full_text", "lang"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.error(f"Missing required columns in dataset: {missing_cols}")
        sys.exit(1)
        
    df = df[required_cols].copy()
    logger.info(f"Loaded {len(df):,} raw tweet entries.")
    
    # ── Step 2: Data Cleaning ───────────────────────────────────
    logger.info("Step 2: Performing initial data cleaning (handling nulls and duplicates)...")
    initial_count = len(df)
    
    # Drop missing full_text
    df = df.dropna(subset=["full_text"]).copy()
    no_nulls_count = len(df)
    
    # Drop duplicates
    df = df.drop_duplicates(subset="full_text", keep="first").reset_index(drop=True)
    clean_count = len(df)
    
    logger.info(f"Initial row count: {initial_count:,}")
    logger.info(f"After dropping missing full_text: {no_nulls_count:,} (Removed: {initial_count - no_nulls_count:,})")
    logger.info(f"After dropping duplicates: {clean_count:,} (Removed: {no_nulls_count - clean_count:,})")
    logger.info(f"Data cleaning complete. Net clean rows: {clean_count:,}")
    
    # ── Step 3: Text Preprocessing ─────────────────────────────
    logger.info("Step 3: Instantiating TextPreprocessor and preprocessing tweet contents...")
    preprocessor = TextPreprocessor()
    
    # Apply pipeline to dataframe (shows progress/log implicitly)
    df["processed_text"] = df["full_text"].apply(preprocessor.preprocess)
    
    # Filter out empty texts after preprocessing
    before_empty_filter = len(df)
    df = df[df["processed_text"].str.strip().str.len() > 0].reset_index(drop=True)
    after_empty_filter = len(df)
    
    logger.info(f"After preprocessing: {after_empty_filter:,} rows (Removed {before_empty_filter - after_empty_filter:,} non-Indonesian or empty entries).")
    
    # ── Step 4: Lexicon Labeling ────────────────────────────────
    logger.info("Step 4: Executing Lexicon-based sentiment labeling (InSet Fajri et al.)...")
    labeler = LexiconLabeler()
    
    # Compute scores and classes
    lexicon_results = [labeler.label_sentiment(t) for t in df["processed_text"]]
    df["Score"] = [r[0] for r in lexicon_results]
    df["label"] = [r[1] for r in lexicon_results]
    
    # Print label distribution
    dist = df["label"].value_counts()
    logger.info("Lexicon labeling completed. Sentiment Class Distribution:")
    for kls, jml in dist.items():
        pct = (jml / len(df)) * 100
        logger.info(f"  - {kls:<8}: {jml:>6,} ({pct:>5.1f}%)")
        
    # ── Step 5: Exploratory Data Analysis ───────────────────────
    logger.info(f"Step 5: Generating and saving EDA plots and reports to {reports_dir}...")
    run_all_eda(df, reports_dir)
    
    # ── Step 6: Model Training & Evaluation ─────────────────────
    logger.info("Step 6: Splitting datasets, training models, and tuning hyperparameters...")
    run_tuning = not skip_tuning
    model, tfidf, le, metadata = train_pipeline(
        df=df,
        model_dir=model_dir,
        reports_dir=reports_dir,
        run_tuning=run_tuning,
        random_state=42
    )
    
    # ── Step 7: Verification / Test Predict ────────────────────
    if run_test:
        logger.info("Step 7: Verifying model behavior with inference test cases...")
        try:
            predictor = SentimentPredictor(model_dir=model_dir)
            test_sentences = [
                # ── Contoh 1 : Sentimen Positif ──────────────────────────
                "Alhamdulillah akhirnya lolos LPDP! Bangga banget, "
                "programnya bener-bener membantu anak bangsa buat lanjut S2 ke luar negeri. "
                "Terima kasih LPDP, semangat buat yang masih berjuang!",

                # ── Contoh 2 : Sentimen Negatif ──────────────────────────
                "Proses seleksi LPDP sangat mengecewakan, dokumen adminnya ribet banget "
                "dan syaratnya sering berubah tanpa pemberitahuan yang jelas. "
                "Banyak pelamar berprestasi yang gagal hanya karena birokrasi amburadul.",

                # ── Contoh 3 : Sentimen Netral ───────────────────────────
                "LPDP membuka pendaftaran beasiswa reguler dalam negeri dan luar negeri. "
                "Kuota yang tersedia tahun ini sebanyak 4.000 awardee. "
                "Batas waktu pendaftaran adalah 30 Juni, persyaratan lengkap di web resmi LPDP.",
            ]
            
            logger.info("=" * 60)
            logger.info("TEST CASE RESULTS")
            logger.info("=" * 60)
            for i, sent in enumerate(test_sentences, 1):
                res = predictor.predict(sent)
                logger.info(f"Test #{i}:")
                logger.info(f"  📝 Input  : {res['text_original']}")
                logger.info(f"  ⚙️ Clean  : {res['text_preprocessed']}")
                logger.info(f"  🎯 Class  : {res['sentiment']} (Confidence: {res['confidence']:.2%})")
            logger.info("=" * 60)
        except Exception as e:
            logger.error(f"Inference validation failed: {e}")
            
    end_time = datetime.now()
    duration = end_time - start_time
    logger.info(f"🎉 Pipeline successfully finished in: {duration}")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="LPDP Twitter Sentiment Analysis Pipeline")
    parser.add_argument(
        "--dataset-url",
        type=str,
        default="https://raw.githubusercontent.com/go0se05/Analysis-Sentiment_LPDP/refs/heads/main/master_data_lpdp.csv",
        help="URL to fetch the dataset from (default: GitHub Master Data CSV)"
    )
    parser.add_argument(
        "--skip-tuning",
        action="store_true",
        help="If set, skips the randomized search hyperparameter tuning to save time"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="saved_model",
        help="Directory to save model artifacts (default: 'saved_model')"
    )
    parser.add_argument(
        "--reports-dir",
        type=str,
        default="reports",
        help="Directory to save output reports and figures (default: 'reports')"
    )
    parser.add_argument(
        "--no-test",
        action="store_true",
        help="If set, skips the sample inference testing step at the end"
    )
    
    args = parser.parse_args()
    
    run_pipeline(
        dataset_url=args.dataset_url,
        skip_tuning=args.skip_tuning,
        model_dir=args.model_dir,
        reports_dir=args.reports_dir,
        run_test=not args.no_test
    )


if __name__ == "__main__":
    main()
