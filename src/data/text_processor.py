import pandas as pd
from transformers import BertTokenizer, BertModel
import torch

class TextProcessor:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.model = BertModel.from_pretrained("bert-base-uncased")

    def preprocess_text(self, text):
        tokens = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            embeddings = self.model(**tokens).last_hidden_state.mean(dim=1)
        return embeddings

    def process_dataset(self, file_path):
        df = pd.read_csv(file_path)
        df["embeddings"] = df["text"].apply(self.preprocess_text)
        return df

if __name__ == "__main__":
    tp = TextProcessor()
    processed_df = tp.process_dataset("data/raw/text/reddit.csv")
    processed_df.to_pickle("data/processed/text/processed_text.pkl")
