import re
import string

STOPWORDS = {
    'i','me','my','myself','we','our','ours','ourselves','you','your','yours',
    'yourself','yourselves','he','him','his','himself','she','her','hers','herself',
    'it','its','itself','they','them','their','theirs','themselves','what','which',
    'who','whom','this','that','these','those','am','is','are','was','were','be','been',
    'being','have','has','had','having','do','does','did','doing','a','an','the','and',
    'but','if','or','because','as','until','while','of','at','by','for','with','about',
    'against','between','into','through','during','before','after','above','below','to',
    'from','up','down','in','out','on','off','over','under','again','further','then',
    'once','here','there','when','where','why','how','all','any','both','each','few',
    'more','most','other','some','such','no','nor','not','only','own','same','so',
    'than','too','very','s','t','can','will','just','don','should','now'
}

class Preprocessor:
    def __init__(self, texts):
        self.texts = texts
        self.processed = []

    def clean_text(self, text):
        text = text.lower()
        text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
        return text

    def tokenize(self, text):
        return text.split()

    def remove_stopwords(self, tokens):
        return [t for t in tokens if t not in STOPWORDS]

    def preprocess(self):
        self.processed = []
        for text in self.texts:
            cleaned = self.clean_text(text)
            tokens = self.tokenize(cleaned)
            tokens = self.remove_stopwords(tokens)
            self.processed.append(tokens)

    def get_processed(self):
        return self.processed


# Example usage
from .data_load import DataSource

source = DataSource('/Users/jenishshekhada/Desktop/Inten/dynamic-ai-customer-support/backend /data/training_data.txt')
source.load_data()
texts = source.get_data()

processor = Preprocessor(texts)
processor.preprocess()
processed_texts = processor.get_processed()

print("\n")
print("----------------------------------")
print("Processed Texts:")
print("----------------------------------")
print("\n")
for pt in processed_texts:
    print(pt)
