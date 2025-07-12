import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords

nltk.download("punkt")
nltk.download("stopwords")

# NLP Tokenizer & Preprocessing
stemmer = SnowballStemmer(language="english")

def tokenize(text):
    """Tokenize and stem text for NLP processing"""
    return [stemmer.stem(token) for token in word_tokenize(text)]

english_stopwords = stopwords.words("english")

# Vectorizer for text feature extraction
def vectorizer():
    return TfidfVectorizer(tokenizer=tokenize, stop_words=english_stopwords)
