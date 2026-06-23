import logging
from typing import Dict, Tuple, Set
import requests
import pandas as pd
from io import StringIO

logger = logging.getLogger(__name__)

# Fallback basic lexicons for safety if URL fetching is blocked/offline
DEFAULT_POS_LEXICON = {
    "bagus": 3, "senang": 4, "hebat": 4, "cinta": 5, "setuju": 3, "baik": 3,
    "lolos": 4, "sukses": 4, "semangat": 4, "terima kasih": 5, "membantu": 3,
    "bangga": 4, "alhamdulillah": 5, "untung": 3, "mudah": 3, "menang": 4,
}

DEFAULT_NEG_LEXICON = {
    "jelek": -3, "marah": -4, "benci": -5, "kecewa": -4, "gagal": -4,
    "susah": -3, "ribet": -3, "buruk": -3, "rugi": -3, "lemah": -2,
    "mahal": -2, "salah": -2, "mengecewakan": -5, "amburadul": -4,
}


class LexiconLabeler:
    """Performs lexicon-based sentiment analysis for Indonesian text.
    
    Downloads or uses fallback sentiment lexicons (InSet Fajri et al.) to compute
    sentiment score:
        score = sum(positive_weights) - sum(abs(negative_weights))
    And assigns a categorical label:
        score > 0  -> Positif
        score < 0  -> Negatif
        score == 0 -> Netral
    """
    
    def __init__(
        self,
        pos_url: str = "https://raw.githubusercontent.com/fajri91/InSet/master/positive.tsv",
        neg_url: str = "https://raw.githubusercontent.com/fajri91/InSet/master/negative.tsv"
    ):
        """Initializes the lexicon sets and resolves conflicts."""
        logger.info("Initializing LexiconLabeler...")
        
        self.lexicon_pos: Dict[str, int] = self._load_tsv_lexicon(pos_url, is_positive=True)
        self.lexicon_neg: Dict[str, int] = self._load_tsv_lexicon(neg_url, is_positive=False)
        
        # Resolve conflicts: words that appear in both positive and negative lists
        self._resolve_conflicts()
        
    def _load_tsv_lexicon(self, url: str, is_positive: bool) -> Dict[str, int]:
        """Loads a TSV lexicon from a URL with custom formatting and fallback."""
        try:
            logger.info(f"Loading lexicon from {url}...")
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            
            df = pd.read_csv(
                StringIO(r.text),
                sep="\t",
                header=None,
                names=["word", "weight"]
            )
            
            # Formatting and cleaning
            df["word"] = df["word"].astype(str).str.lower().str.strip()
            df["weight"] = pd.to_numeric(df["weight"], errors="coerce")
            df = df.dropna(subset=["weight"])
            df["weight"] = df["weight"].astype(int)
            
            mapping = dict(zip(df["word"], df["weight"]))
            logger.info(f"Loaded {len(mapping)} words from {url}.")
            return mapping
            
        except Exception as e:
            logger.warning(f"Failed to load lexicon from URL ({e}). Using local fallback.")
            return DEFAULT_POS_LEXICON if is_positive else DEFAULT_NEG_LEXICON
            
    def _resolve_conflicts(self) -> None:
        """Resolves words that appear in both positive and negative lexicons
        by keeping only the one with the higher absolute weight (netted out),
        and applies explicit overrides for sentiment accuracy.
        """
        conflicts: Set[str] = set(self.lexicon_pos.keys()) & set(self.lexicon_neg.keys())
        if conflicts:
            logger.info(f"Resolving {len(conflicts)} conflicting words appearing in both positive and negative lexicons...")
            for word in conflicts:
                pos_val = self.lexicon_pos[word]
                neg_val = abs(self.lexicon_neg[word])
                if pos_val > neg_val:
                    self.lexicon_pos[word] = pos_val - neg_val
                    self.lexicon_neg.pop(word, None)
                elif neg_val > pos_val:
                    self.lexicon_neg[word] = -(neg_val - pos_val)
                    self.lexicon_pos.pop(word, None)
                else:
                    self.lexicon_pos.pop(word, None)
                    self.lexicon_neg.pop(word, None)
            logger.info("Conflict resolution finished.")

        # Explicit overrides to fix lexicon noise, translation errors, and domain bias
        lexicon_overrides = {
            # Positive words
            "lolos": 4, "bantu": 4, "membantu": 4, "benar": 2, "bangga": 4, "alhamdulillah": 5,
            "bagus": 4, "sukses": 4, "hebat": 4, "cinta": 5, "setuju": 3, "baik": 3,
            "semangat": 4, "terimakasih": 5, "untung": 3, "mudah": 3, "menang": 4,
            "bahagia": 4, "cocok": 3, "kuat": 3, "luar biasa": 4, "senang": 4,
            "gembira": 4, "puji": 3, "bersyukur": 5, "aman": 3, "damai": 3, "maju": 3,
            "pintar": 3, "cerdas": 4, "kreatif": 3,
            
            # Negative words
            "kecewa": -4, "mengecewakan": -5, "ribet": -3, "buruk": -4, "jelek": -3,
            "gagal": -4, "susah": -3, "sulit": -3, "rugi": -3, "lemah": -2, "mahal": -2,
            "salah": -2, "benci": -5, "marah": -4, "amburadul": -4, "parah": -3,
            "payah": -3, "hancur": -4, "kacau": -4, "masalah": -3, "permasalahan": -3,
            "tuntut": -2, "tuntutan": -2, "sesal": -3, "menyesal": -3,
            
            # Neutral overrides (forced to 0 to prevent sentiment classification bias)
            "cukup": 0, "anak": 0, "jangan": 0, "tidak": 0, "gak": 0, "ga": 0, "nggak": 0,
            "apa": 0, "semua": 0, "ikut": 0, "kalau": 0, "kalo": 0, "kayak": 0, "pilih": 0,
            "kata": 0, "masuk": 0, "pajak": 0, "sama": 0, "punya": 0, "buat": 0, "bikin": 0,
            "laku": 0, "lakukan": 0, "ada": 0, "adalah": 0, "yaitu": 0, "merupakan": 0,
            "menjadi": 0, "jadi": 0, "bisa": 0, "dapat": 0, "luar": 0, "negeri": 0,
            "dalam": 0, "negara": 0, "indonesia": 0, "warga": 0, "rakyat": 0, "lpdp": 0,
            "beasiswa": 0, "awardee": 0, "penerima": 0, "penerimabeasiswa": 0, "pendaftaran": 0,
            "daftar": 0, "seleksi": 0, "syarat": 0, "reguler": 0, "kuota": 0, "kuliah": 0,
            "sekolah": 0, "studi": 0, "belajar": 0, "pendidikan": 0, "didik": 0, "universitas": 0,
            "kampus": 0, "uras": 0, "brain": 0, "drain": 0, "alumni": 0, "lulusan": 0,
            "video": 0, "unggah": 0, "posting": 0, "cuitan": 0, "tweet": 0, "netizen": 0,
            "warganet": 0, "bapak": 0, "ibu": 0, "orang": 0, "orangtua": 0, "keluarga": 0,
            "tahun": 0, "bulan": 0, "hari": 0, "jam": 0, "tanggal": 0, "juni": 0, "juli": 0,
            "waktu": 0, "kali": 0, "dulu": 0, "sekarang": 0, "nanti": 0, "besok": 0,
            "kemarin": 0, "lusa": 0, "info": 0, "bilang": 0, "lanjut": 0, "akhir": 0, "batas": 0,
            "sedia": 0, "lengkap": 0, "resmi": 0, "buka": 0, "banyak": 0, "proses": 0, "sangat": 0,
            "sering": 0, "pemberitahuan": 0, "jelas": 0, "dokumen": 0, "admin": 0, "adminnya": 0,
            "ubah": 0, "berubah": 0, "pelamar": 0, "lamar": 0, "birokrasi": 0
        }
        
        logger.info(f"Applying {len(lexicon_overrides)} custom lexicon overrides...")
        for word, weight in lexicon_overrides.items():
            if weight > 0:
                self.lexicon_pos[word] = weight
                self.lexicon_neg.pop(word, None)
            elif weight < 0:
                self.lexicon_neg[word] = weight
                self.lexicon_pos.pop(word, None)
            else:
                self.lexicon_pos.pop(word, None)
                self.lexicon_neg.pop(word, None)
        logger.info("Custom lexicon overrides applied.")
            
    def label_sentiment(self, text: str) -> Tuple[int, str]:
        """Calculates the sentiment score and determines the label class.
        
        Args:
            text: A preprocessed string of text.
            
        Returns:
            Tuple[int, str]: (sentiment_score, sentiment_label)
        """
        if not isinstance(text, str) or not text.strip():
            return 0, "Netral"
            
        score = 0
        words = text.split()
        
        for token in words:
            score += self.lexicon_pos.get(token, 0)
            score -= abs(self.lexicon_neg.get(token, 0))
            
        if score > 0:
            return score, "Positif"
        elif score < 0:
            return score, "Negatif"
        else:
            return 0, "Netral"
