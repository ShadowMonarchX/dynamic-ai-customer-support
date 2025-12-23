from Data_Load import DataSource
import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

class Preprocessor:
    def __init__(self, texts):
        self.texts = texts
        self.processed = []

    def clean_text(self, text):
        text = text.lower()
        text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
        return text

    def tokenize(self, text):
        return word_tokenize(text)

    def remove_stopwords(self, tokens):
        stop_words = set(stopwords.words("english"))
        return [t for t in tokens if t not in stop_words]

    def normalize(self, tokens):
        return [t.lower() for t in tokens]

    def preprocess(self):
        self.processed = []
        for text in self.texts:
            cleaned = self.clean_text(text)
            tokens = self.tokenize(cleaned)
            tokens = self.remove_stopwords(tokens)
            tokens = self.normalize(tokens)
            self.processed.append(tokens)

    def get_processed(self):
        return self.processed


source = DataSource("data")
source.load_data()
texts = source.get_data()

processor = Preprocessor(texts)
processor.preprocess()
processed_texts = processor.get_processed()

for pt in processed_texts:
    print(pt)
