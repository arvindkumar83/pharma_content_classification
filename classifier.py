import os, re, json, uuid, hashlib
from pathlib import Path
import pdfplumber
import pandas as pd
import psycopg2
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

from huggingface_hub import login

login("YOUR_HUGGUNGFACE_API_KEY")


####################################################
# Step 0: Ontology definition
####################################################
ONTOLOGY = {
    "labels": [
        "Clinical Trials", "Financials", "Market Share", "Geography", "Milestones",
        "Sales Output", "Regulatory", "Pipeline/Dev Status", "Safety", "Efficacy",
        "Partnerships/Deals", "IP/Patents", "Other"
    ],
    "entities": [
        "COMPANY","PRODUCT","INDICATION","TRIAL_ID","DATE","CURRENCY","PERCENT","NUMBER","REGION","REGULATOR"
    ]
}

####################################################
# Step 1: Ingestion/parsing (PDF only demo)
####################################################
def ingest_pdf(path):
    doc_id = str(uuid.uuid4())
    chunks = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ''
            chunks.append({"doc_id": doc_id, "page": i, "text": text})
    return {"doc_id": doc_id, "chunks": chunks}

####################################################
# Step 2: Chunking
####################################################
def chunk_text(chunks, max_len=800):
    out = []
    for ch in chunks:
        words = ch["text"].split()
        for i in range(0, len(words), max_len):
            seg = " ".join(words[i:i+max_len])
            out.append({"doc_id": ch["doc_id"], "page": ch["page"], "text": seg})
    return out

####################################################
# Step 3: Weak labeling with regex/keywords
####################################################
KEYWORDS = {
    "Financials": [r"revenue", r"EBITDA", r"profit", r"loss", r"USD", r"%"],
    "Clinical Trials": [r"Phase [I|II|III|IV]", r"NCT[0-9]{8}"],
    "Market Share": [r"market share", r"CAGR"],
    "Regulatory": [r"FDA", r"EMA", r"NDA", r"approval"],
}

def weak_label(text):
    labels = []
    for lab, pats in KEYWORDS.items():
        for p in pats:
            if re.search(p, text, re.I):
                labels.append(lab)
                break
    return list(set(labels)) or ["Other"]

####################################################
# Step 4: Annotation (stub)
####################################################
def manual_annotation_interface():
    print("Use Doccano/Label Studio externally. Stub only.")

####################################################
# Step 5: Multi-label classification (fine-tune placeholder)
####################################################
labels = ONTOLOGY["labels"]
tok = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")
mdl = AutoModelForSequenceClassification.from_pretrained(
    "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
    num_labels=len(labels), problem_type="multi_label_classification"
)

def classify_chunks(texts, threshold=0.35):
    batch = tok(texts, truncation=True, padding=True, return_tensors="pt")
    with torch.no_grad():
        logits = mdl(**batch).logits
        probs = torch.sigmoid(logits).cpu().numpy()
    return [{labels[i]: float(p[i]) for i in (p >= threshold).nonzero()[0]} for p in probs]

####################################################
# Step 6: NER + linking (regex + stub)
####################################################
ENTITY_PATTERNS = {
    "TRIAL_ID": r"NCT[0-9]{8}",
    "CURRENCY": r"\$[0-9,.]+|USD|EUR|INR",
    "PERCENT": r"[0-9]+(\.[0-9]+)?%",
    "NUMBER": r"[0-9,.]+",
    "DATE": r"\b(19|20)[0-9]{2}\b",
}

def extract_entities(text):
    ents = []
    for typ, pat in ENTITY_PATTERNS.items():
        for m in re.finditer(pat, text):
            ents.append({"type": typ, "surface": m.group(), "span": (m.start(), m.end())})
    return ents

####################################################
# Step 7: Numeric extraction (finance/trials)
####################################################
def extract_metrics(text):
    metrics = []
    for cur in re.findall(ENTITY_PATTERNS["CURRENCY"], text):
        metrics.append({"name": "financial", "value": cur})
    return metrics

####################################################
# Step 8: Embeddings + pgvector upsert
####################################################
EMB_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def embed_chunks(chunks):
    texts = [c["text"] for c in chunks]
    vecs = EMB_MODEL.encode(texts, normalize_embeddings=True)
    for c, v in zip(chunks, vecs):
        c["embedding"] = v.tolist()
    return chunks

####################################################
# Step 9: RAG Retriever + simple QA
####################################################
def retrieve(query, chunks, topk=3):
    q_emb = EMB_MODEL.encode([query], normalize_embeddings=True)[0]
    sims = [(c, float(sum(a*b for a,b in zip(q_emb, c["embedding"])))) for c in chunks]
    sims = sorted(sims, key=lambda x: x[1], reverse=True)
    return sims[:topk]

####################################################
# Step 10: Orchestration (demo flow)
####################################################
def pipeline(path, query):
    doc = ingest_pdf(path)
    chs = chunk_text(doc["chunks"])
    for c in chs:
        c["weak_labels"] = weak_label(c["text"])
        c["entities"] = extract_entities(c["text"])
        c["metrics"] = extract_metrics(c["text"])
    chs = embed_chunks(chs)
    results = retrieve(query, chs)
    return results

####################################################
# Step 11: Evaluation (stub)
####################################################
def evaluate(pred_labels, gold_labels):
    correct = sum(1 for p,g in zip(pred_labels,gold_labels) if set(p)==set(g))
    return correct/len(gold_labels)



if __name__ == "__main__":
    pdf_path = "sample.pdf"  # replace with real file
    if Path(pdf_path).exists():
        hits = pipeline(pdf_path, "The acquisition of Dicerna Pharmaceuticals and their RNAi")
        for h,score in hits:
            print("SCORE", score, "TEXT", h["text"])
    else:
        print("Put a sample.pdf in the folder to demo.")
