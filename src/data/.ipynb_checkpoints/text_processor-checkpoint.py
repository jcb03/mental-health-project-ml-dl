from transformers import BertTokenizer

def preprocess_text(text):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    return tokenizer(text, return_tensors="pt", padding=True, truncation=True)
