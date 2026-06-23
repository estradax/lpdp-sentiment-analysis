import logging
import re
from typing import Dict, Set
import requests
import pandas as pd
from langdetect import detect, LangDetectException
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# --- Performance Monkey-Patch for Sastrawi ArrayDictionary ---
from Sastrawi.Dictionary.ArrayDictionary import ArrayDictionary

def _patched_init(self, words=None):
    self.words = set()
    if words:
        self.add_words(words)

def _patched_contains(self, word):
    if not word or word.strip() == '':
        return False
    return word in self.words

def _patched_count(self):
    return len(self.words)

def _patched_add_words(self, words):
    clean_words = {w for w in words if w and w.strip() != ''}
    self.words.update(clean_words)

def _patched_add(self, word):
    if not word or word.strip() == '':
        return
    self.words.add(word)

ArrayDictionary.__init__ = _patched_init
ArrayDictionary.contains = _patched_contains
ArrayDictionary.count = _patched_count
ArrayDictionary.add_words = _patched_add_words
ArrayDictionary.add = _patched_add
# -------------------------------------------------------------

logger = logging.getLogger(__name__)

# Strong Indonesian words for language detection heuristic
STRONG_ID_WORDS = {
    "yg", "dgn", "nya", "jg", "juga", "ini", "itu", "dan", "atau", "untuk",
    "dengan", "ada", "tidak", "bisa", "aja", "gak", "ga", "gw", "aku",
    "kamu", "saya", "kalian", "kita", "mereka", "dia", "udah", "sudah",
    "jadi", "sih", "deh", "nih", "lah", "dong", "banget", "bgt"
}

# Fallback normalization dictionary (offline safety)
DEFAULT_KAMUS_BAKU = {
    "gak": "tidak", "ga": "tidak", "udah": "sudah",
    "gimana": "bagaimana", "kalo": "kalau", "aja": "saja",
    "bgt": "banget", "yg": "yang", "dgn": "dengan",
    "tdk": "tidak", "blm": "belum", "sdh": "sudah",
    "krn": "karena", "utk": "untuk", "lg": "lagi",
    "emg": "memang", "ttp": "tetap", "hrs": "harus",
    "pake": "pakai", "pengen": "ingin", "kmrn": "kemarin",
    "abis": "habis", "bikin": "membuat", "nggak": "tidak",
}

# Custom stop words specifically for Indonesian Twitter context
CUSTOM_STOPWORDS = {
    "yg", "dgn", "nya", "jg", "juga", "ini", "itu", "dan", "atau", "untuk",
    "dengan", "ada", "tidak", "bisa", "aja", "gak", "ga", "gw", "aku",
    "kamu", "saya", "kalian", "kita", "mereka", "dia", "udah", "sudah",
    "jadi", "ya", "sih", "deh", "nih", "lah", "dong", "banget", "bgt",
    "wkwk", "haha", "lol", "wah", "oh", "ah", "eh", "hm",
    "rt", "amp", "via", "cc", "https", "http", "co", "www",
}


class TextPreprocessor:
    """Handles Indonesian text cleaning and NLP preprocessing.
    
    Processes text through a series of stages:
    1. Filter: remove URLs, mentions, emojis, numbers, special characters.
    2. Case folding: lowercase.
    3. Language detection: keep only Indonesian text.
    4. Word normalization: correct alay/slang words using a baking dictionary.
    5. Stopwords removal: remove common words (Sastrawi + custom Twitter list).
    6. Stemming: reduce words to their morphological base form.
    """
    
    def __init__(self, kamus_url: str = "https://raw.githubusercontent.com/go0se05/PI/main/kamuskatabaku.csv"):
        """Initializes preprocessor components and loads dictionary mapping."""
        logger.info("Initializing preprocessor components...")
        
        # 1. Stemmer Sastrawi
        stemmer_factory = StemmerFactory()
        self.stemmer = stemmer_factory.create_stemmer()
        
        # 2. Stopwords set
        sw_factory = StopWordRemoverFactory()
        self.stopwords_set: Set[str] = set(sw_factory.get_stop_words())
        self.stopwords_set.update(CUSTOM_STOPWORDS)
        
        # 3. Load normalization dictionary (with fallback)
        self.kamus_baku: Dict[str, str] = self._load_normalization_dict(kamus_url)
        
        # Precompile regex patterns for efficiency
        self.url_pattern = re.compile(r"http\S+|www\S+")
        self.mention_pattern = re.compile(r"@\w+")
        self.hashtag_pattern = re.compile(r"#(\w+)")
        self.emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"
            "\U0001F300-\U0001F5FF"
            "\U0001F680-\U0001F6FF"
            "\U0001F700-\U0001F77F"
            "\U0001F780-\U0001F7FF"
            "\U0001F800-\U0001F8FF"
            "\U0001F900-\U0001F9FF"
            "\U0001FA00-\U0001FA6F"
            "\U0001FA70-\U0001FAFF"
            "\U00002702-\U000027B0"
            "]+",
            flags=re.UNICODE,
        )
        self.digits_pattern = re.compile(r"\d+")
        self.non_alpha_pattern = re.compile(r"[^\w\s]")
        self.non_alpha_pattern = re.compile(r"[^\w\s]")
        self.underscores_pattern = re.compile(r"_+")
        self.spaces_pattern = re.compile(r"\s+")

    def _load_normalization_dict(self, url: str) -> Dict[str, str]:
        """Loads alay-to-formal word dictionary from URL or local fallback."""
        try:
            logger.info(f"Loading normalization dictionary from {url}...")
            # Set a reasonable timeout so script doesn't hang offline
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            
            # Read CSV structure
            from io import StringIO
            df = pd.read_csv(StringIO(r.text))
            
            cols = df.columns.str.lower().tolist()
            if "tidak_baku" in cols and "kata_baku" in cols:
                mapping = dict(zip(df["tidak_baku"], df["kata_baku"]))
            else:
                mapping = dict(zip(df.iloc[:, 0], df.iloc[:, 1]))
                
            logger.info(f"Loaded {len(mapping)} normalization rules successfully.")
            return mapping
            
        except Exception as e:
            logger.warning(f"Failed to load dictionary from URL ({e}). Using local fallback dictionary.")
            return DEFAULT_KAMUS_BAKU

    def filter_text(self, text: str) -> str:
        """Removes URLs, mentions, emojis, numbers, and cleans whitespace."""
        if not isinstance(text, str):
            return ""
            
        # Clean URLs, Mentions, and Hashtag symbols (keep hashtag word)
        text = self.url_pattern.sub(" ", text)
        text = self.mention_pattern.sub(" ", text)
        text = self.hashtag_pattern.sub(r"\1", text)
        
        # Clean Emojis, digits, and special characters
        text = self.emoji_pattern.sub(" ", text)
        text = self.digits_pattern.sub(" ", text)
        text = self.non_alpha_pattern.sub(" ", text)
        text = self.underscores_pattern.sub(" ", text)
        
        # Clean spaces
        text = self.spaces_pattern.sub(" ", text).strip()
        return text

    def is_indonesian(self, text: str) -> bool:
        """Heuristic and package-based language checking.
        
        Treated as Indonesian if the text contains fewer than 3 words,
        or if it contains at least 2 strong Indonesian words (fast-path),
        or if langdetect categorizes it as 'id'.
        """
        words = text.split()
        if len(words) < 3:
            return True
            
        # Fast path heuristic to bypass slow langdetect for obvious Indonesian texts
        id_words_count = sum(1 for w in words if w in STRONG_ID_WORDS)
        if id_words_count >= 2:
            return True
            
        try:
            return detect(text) == "id"
        except LangDetectException:
            return True

    def normalize_words(self, text: str) -> str:
        """Replaces slang/alay words with their formal counterparts."""
        words = text.split()
        normalized = [self.kamus_baku.get(w, w) for w in words]
        return " ".join(normalized)

    def remove_stopwords(self, text: str) -> str:
        """Filters out words in stopwords set. Keep words with len > 1."""
        words = text.split()
        filtered = [w for w in words if w not in self.stopwords_set and len(w) > 1]
        return " ".join(filtered)

    def stem(self, text: str) -> str:
        """Stems Indonesian words to base form using Sastrawi."""
        if not text.strip():
            return ""
        return self.stemmer.stem(text)

    def preprocess(self, text: str, filter_lang: bool = True) -> str:
        """Executes the full preprocessing pipeline on a string.
        
        Returns empty string if language filter is enabled and text
        is not in Indonesian, or if text is empty.
        """
        text_clean = self.filter_text(text)
        if not text_clean:
            return ""
            
        text_lower = text_clean.lower()
        
        if filter_lang and not self.is_indonesian(text_lower):
            return ""
            
        text_norm = self.normalize_words(text_lower)
        text_sw = self.remove_stopwords(text_norm)
        text_stem = self.stem(text_sw)
        
        return text_stem
