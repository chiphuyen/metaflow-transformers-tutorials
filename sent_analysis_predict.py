import torch

from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

BASE_MODEL = "distilbert-base-uncased"
CKPT_PATH = "checkpoints/saved_model_v0"

def predict(ckpt_path, review):
    model = DistilBertForSequenceClassification.from_pretrained(ckpt_path)
    model.eval()
    tokenizer = DistilBertTokenizerFast.from_pretrained(BASE_MODEL)
    encoding = tokenizer(review, padding=True, truncation=True, return_tensors="pt")
    output = model(**encoding)
    probs = torch.nn.functional.softmax(output["logits"], dim=1)
    pred = torch.argmax(probs).numpy()
    sentiment = "POSITIVE" if pred == 1 else "NEGATIVE"
    return sentiment


predict(CKPT_PATH, "this movie is awesome")