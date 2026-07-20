import sys
import os
import json

# Add project root directory to python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.inference import SentimentPredictor, BertSentimentPredictor


def verify_model_inference(model_name: str, predictor_class, default_dir: str, test_sentences, expected_labels) -> bool:
    print(f"\n======================================================================")
    print(f"🔮 RUNNING {model_name.upper()} INFERENCE VERIFICATION")
    print(f"======================================================================")

    if not os.path.exists(default_dir):
        print(f"❌ Error: Model directory '{default_dir}' not found. Skipping {model_name} verification.")
        return False

    print(f"Initializing {predictor_class.__name__} from '{default_dir}'...")
    try:
        predictor = predictor_class(model_dir=default_dir)
    except Exception as e:
        print(f"❌ Failed to load predictor for {model_name}: {e}")
        return False

    passed_all = True
    for i, (sent, expected) in enumerate(zip(test_sentences, expected_labels), 1):
        res = predictor.predict(sent)
        pred_label = res["sentiment"]
        confidence = res["confidence"]

        status = "PASSED" if pred_label == expected else "FAILED"
        if pred_label != expected:
            passed_all = False

        print(f"\nContoh #{i}:")
        print(f"  📝 Input    : {res['text_original']}")
        print(f"  🎯 Expected : {expected}")
        print(f"  🏷️ Predicted: {pred_label} (Confidence: {confidence:.2%})")
        print(f"  ✅ Status   : {status}")
        print("-" * 70)

    if passed_all:
        print(f"\n🎉 SUCCESS: All {model_name} test cases passed!")
    else:
        print(f"\n⚠️ WARNING: Some {model_name} test cases did not match expected labels.")

    return passed_all


def main():
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

        # ── Contoh 4 : Kalimat dengan negasi ─────────────────────
        "Saya tidak pernah kecewa sama sekali dengan LPDP, semua prosesnya jelas dan membantu.",
    ]

    expected_labels = ["Positif", "Negatif", "Netral", "Positif"]

    rf_passed = verify_model_inference(
        model_name="Random Forest",
        predictor_class=SentimentPredictor,
        default_dir=os.path.join(project_root, "weights", "random-forest"),
        test_sentences=test_sentences,
        expected_labels=expected_labels
    )

    bert_passed = verify_model_inference(
        model_name="IndoBERT",
        predictor_class=BertSentimentPredictor,
        default_dir=os.path.join(project_root, "weights", "indobert"),
        test_sentences=test_sentences,
        expected_labels=expected_labels
    )

    print("\n" + "=" * 70)
    print("📊 OVERALL VERIFICATION SUMMARY:")
    print(f"  - Random Forest: {'✅ PASSED' if rf_passed else '❌ FAILED'}")
    print(f"  - IndoBERT     : {'✅ PASSED' if bert_passed else '❌ FAILED'}")
    print("=" * 70)

    if not (rf_passed and bert_passed):
        sys.exit(1)


if __name__ == "__main__":
    main()
