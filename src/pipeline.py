import os
import joblib
import spacy
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict, Any
from pydantic import BaseModel
from fastapi import HTTPException

TFIDF_PIPELINE_PATH = '../models/tfidf_svc_pipeline(best).joblib'
TRANSFORMER_MODEL_DIR = '../models/bert'
LABEL_MAP_PATH = '../models/label_map.joblib'

baseline_pipeline = joblib.load('../models/tfidf_svc_pipeline(best).joblib')

tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL_DIR)
transformer = AutoModelForSequenceClassification.from_pretrained(TRANSFORMER_MODEL_DIR)
transformer.eval()

label_map: Dict[int, str] = joblib.load(LABEL_MAP_PATH)

nlp = spacy.load("ru_core_news_sm", disable=["parser", "ner"])


def spacy_tokenize(text: str) -> List[str]:
    """
    Tokenize, lemmatize, remove stop-words and non-alpha tokens.
    Mirrors the TF-IDF preprocessing during training.
    """
    doc = nlp(text.lower())
    return [tok.lemma_ for tok in doc if tok.is_alpha and not tok.is_stop]


class SpacyTokenizerWrapper:
    """
    Callable wrapper for scikit-learn Pipeline.
    """

    def __call__(self, raw_text: str) -> List[str]:
        return spacy_tokenize(raw_text)


def predict_baseline(texts: List[str]) -> List[Dict[str, Any]]:
    """
    Run TF-IDF + classical classifier pipeline.
    Returns list of dicts: {"label": str, "proba": float}
    """
    # Ensure texts are preprocessed via the same tokenizer
    # Pipeline should include TfidfVectorizer(tokenizer=SpacyTokenizerWrapper())
    probs = baseline_pipeline.predict_proba(texts)
    preds = baseline_pipeline.predict(texts)
    results = []
    for i, label_idx in enumerate(preds):
        results.append({
            "label": label_map[label_idx],
            "proba": float(probs[i, label_idx])
        })
    return results


def predict_transformer(texts: List[str], device: str = "cpu") -> List[Dict[str, Any]]:
    """
    Run fine-tuned transformer model.
    Returns list of dicts: {"label": str, "proba": float}
    """
    # Basic normalization: lowercasing
    normalized = [t.lower().strip() for t in texts]
    inputs = tokenizer(
        normalized,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        logits = transformer(**inputs).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        preds = probs.argmax(axis=-1)

    results = []
    for i, label_idx in enumerate(preds):
        results.append({
            "label": label_map[int(label_idx)],
            "proba": float(probs[i, label_idx])
        })
    return results


class PredictRequest(BaseModel):
    text: str
    model: str = "transformer"


class PredictResponse(BaseModel):
    predictions: List[Dict[str, Any]]


def predict(request: PredictRequest) -> PredictResponse:
    if request.model == "baseline":
        preds = predict_baseline([request.text])
    elif request.model == "transformer":
        preds = predict_transformer([request.text])
    else:
        raise HTTPException(status_code=400, detail=f"Unknown model '{request.model}'")
    return PredictResponse(predictions=preds)


if __name__ == "__main__":
    import fire


    def cli(text: str, model: str = "transformer"):
        req = PredictRequest(text=text, model=model)
        resp = predict(req)
        print(resp.json(indent=2, ensure_ascii=False))


    fire.Fire(cli)
