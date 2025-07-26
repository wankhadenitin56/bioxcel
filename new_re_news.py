import os
import time
import re
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer, BertConfig, BertModel
from gliner import GLiNER

# === CONFIGURATION ===
BASE_PATH = r"C:/Users/EZB-VM-06/Downloads"
MODEL_PATH = os.path.join(BASE_PATH, "All_Entities_NoRelation_BIOBERT_Relation_Extraction_model_301k_v1_50epochs.pt")
VOCAB_PATH = os.path.join(BASE_PATH, "vocab.txt")
CONFIG_PATH = os.path.join(BASE_PATH, "bert_config.json")
CSV_PATH = os.path.join(BASE_PATH, "daily_news_report_2025-07-09_402_records (1).csv")
OUTPUT_PATH = os.path.join(BASE_PATH, "GLiNER_with_CustomBioBERT_RE.xlsx")

MAX_LEN = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TAG2IDX = {
    'TARGETS': 0, 'REGULATES': 1, 'TREATS/AS_TREATMENT': 2, 'INHIBITION_INHIBITION': 3,
    'INHIBITION_ACTIVATION': 4, 'MUTATION_CAUSES': 5, 'MUTATION': 6, 'INHIBITION': 7,
    'SYNONYM_OF': 8, 'CAUSES': 9, 'BIOMARKER': 10, 'INTERACT': 11, 'ACTIVATION': 12,
    'ACTIVATION_ACTIVATION': 13, 'INHIBITION_CAUSES': 14, 'ACTIVATION_INHIBITION': 15,
    'ACTIVATION_TREATS': 16, 'INHIBITION_TREATS': 17, 'ACTIVATION_CAUSES': 18,
    'No_Relation': 19
}
IDX2TAG = {v: k for k, v in TAG2IDX.items()}

# === RELATION MODEL ===
class BioBertRE(nn.Module):
    def __init__(self, vocab_len, config, len_tokenizer):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-cased")
        self.bert.resize_token_embeddings(len_tokenizer)
        self.dropout = nn.Dropout(p=0.10)
        self.output = nn.Linear(1536, vocab_len)

    def forward(self, input_ids, attention_mask, e1_e2_start):
        encoded_layer = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        blankv1v2 = encoded_layer[0][:, e1_e2_start, :]

        buffer = []
        for i in range(blankv1v2.shape[0]):
            v1v2 = blankv1v2[i, i, :, :]
            v1v2 = torch.cat((v1v2[0], v1v2[1]))
            buffer.append(v1v2)

        v1v2 = torch.stack(buffer, dim=0)
        out = self.dropout(v1v2)
        out = self.output(out)

        return out, out.argmax(-1)

def get_tokenizer():
    tokenizer = BertTokenizer(vocab_file=VOCAB_PATH, do_lower_case=False)
    tokenizer.add_tokens(['<E1>', '</E1>', '<E2>', '</E2>', '[BLANK]'], special_tokens=True)
    return tokenizer

def load_model(tokenizer):
    config = BertConfig.from_json_file(CONFIG_PATH)
    model = BioBertRE(len(TAG2IDX), config, len(tokenizer))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

def get_e1e2_start(input_ids, e1_id, e2_id):
    try:
        return ([i for i, x in enumerate(input_ids) if x == e1_id][0],
                [i for i, x in enumerate(input_ids) if x == e2_id][0])
    except:
        return (-1, -1)

def predict_relation_biobert(text, drug, disease, tokenizer, model):
    marked_text = text.replace(drug, f"<E1>{drug}</E1>").replace(disease, f"<E2>{disease}</E2>")
    input_ids = tokenizer.encode(marked_text, add_special_tokens=True, max_length=MAX_LEN, truncation=True)
    attention_mask = [1] * len(input_ids)
    input_ids = torch.tensor([input_ids], dtype=torch.long)
    attention_mask = torch.tensor([attention_mask], dtype=torch.long)

    e1_id = tokenizer.convert_tokens_to_ids("<E1>")
    e2_id = tokenizer.convert_tokens_to_ids("<E2>")
    e1_e2_start = get_e1e2_start(input_ids[0].tolist(), e1_id, e2_id)

    if -1 in e1_e2_start:
        return "Invalid Markers"

    input_ids = input_ids.to(DEVICE)
    attention_mask = attention_mask.to(DEVICE)
    e1_e2_tensor = torch.tensor([[e1_e2_start[0], e1_e2_start[1]]], dtype=torch.long).to(DEVICE)

    with torch.no_grad():
        logits, preds = model(input_ids, attention_mask, e1_e2_tensor)
        return IDX2TAG[preds.item()]

# === NER (GLiNER) ===
print("Loading GLiNER...")
gliner_model = GLiNER.from_pretrained("knowledgator/gliner-x-large")
gliner_tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
custom_labels = ["Drug", "Disease", "Organization", "Country"]

def chunk_text(text, tokenizer, max_tokens=512, overlap=20):
    words = text.split()
    chunks, start = [], 0
    while start < len(words):
        for end in range(min(len(words), start + max_tokens), start, -1):
            chunk = " ".join(words[start:end])
            tokens = tokenizer(chunk, return_tensors="pt")
            if tokens.input_ids.shape[1] <= max_tokens:
                chunks.append(chunk)
                break
        start = end - overlap if end - overlap > start else end
    return chunks

# === MAIN PIPELINE ===
df = pd.read_csv(CSV_PATH)
df["drugs"] = ""
df["diseases"] = ""
df["drug_disease_relations"] = ""

print("Loading BioBERT RE model...")
tokenizer = get_tokenizer()
re_model = load_model(tokenizer)

start_time = time.time()
for idx, row in df.iterrows():
    title = str(row.get("Title", ""))
    description = str(row.get("Description", ""))
    full_text = f"{title}\n{description}"

    chunks = chunk_text(full_text, tokenizer)
    all_entities = []
    for chunk in chunks:
        try:
            entities = gliner_model.predict_entities(chunk, custom_labels, threshold=0.70, flat_ner=True)
            all_entities.extend(entities)
        except Exception as e:
            print(f"GLiNER error at row {idx}: {e}")

    drug_list = sorted(set(e["text"] for e in all_entities if e["label"] == "Drug" and e["score"] >= 0.70))
    disease_list = sorted(set(e["text"] for e in all_entities if e["label"] == "Disease" and e["score"] >= 0.70))

    df.at[idx, "drugs"] = ", ".join(drug_list)
    df.at[idx, "diseases"] = ", ".join(disease_list)

    relations = []
    for drug in drug_list:
        for disease in disease_list:
            try:
                label = predict_relation_biobert(full_text, drug, disease, tokenizer, re_model)
                if label != "No_Relation" and label != "Invalid Markers":
                    relations.append(f"{drug} -> {disease} ({label})")
            except Exception as e:
                print(f"RE error at row {idx}, drug={drug}, disease={disease}: {e}")

    df.at[idx, "drug_disease_relations"] = "; ".join(relations)
    print(f"Processed {idx + 1}/{len(df)}")

df.to_excel(OUTPUT_PATH, index=False)
print(f"\n✔ Done! File saved to {OUTPUT_PATH}")
print("⏱ Total time:", round(time.time() - start_time, 2), "seconds")
