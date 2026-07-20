import sys
import os
import json

# Add project root to python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.inference import BertSentimentPredictor

def test_inference():
    model_dir = "weights/indobert"
    if not os.path.exists(model_dir):
        print(f"Error: Model directory '{model_dir}' not found. Please run training first.")
        sys.exit(1)

    print(f"Initializing BertSentimentPredictor from '{model_dir}'...")
    predictor = BertSentimentPredictor(model_dir=model_dir)

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

        # ── Contoh 4 : Kalimat dengan negasi (uji kemampuan BERT memahami konteks) ──
        "Saya tidak pernah kecewa sama sekali dengan LPDP, semua prosesnya jelas dan membantu.",
    ]

    expected_labels = ["Positif", "Negatif", "Netral", "Positif"]

    print("\n🔮 RUNNING INDOBERT INFERENCE VERIFICATION:")
    print("=" * 70)
    
    passed_all = True
    for i, (sent, expected) in enumerate(zip(test_sentences, expected_labels), 1):
        res = predictor.predict(sent)
        pred_label = res["sentiment"]
        confidence = res["confidence"]
        
        status = "PASSED" if pred_label == expected else "FAILED"
        if pred_label != expected:
            passed_all = False
            
        print(f"Contoh #{i}:")
        print(f"  📝 Input    : {res['text_original']}")
        print(f"  🎯 Expected : {expected}")
        print(f"  🏷️ Predicted: {pred_label} (Confidence: {confidence:.2%})")
        print(f"  ✅ Status   : {status}")
        print("-" * 70)
        
    if passed_all:
        print("🎉 SUCCESS: All test cases passed!")
    else:
        print("❌ FAILURE: Some test cases failed. Retraining with corrected data may be needed.")

if __name__ == "__main__":
    test_inference()
