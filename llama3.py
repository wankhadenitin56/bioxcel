import pandas as pd
import requests

OLLAMA_MODEL = "llama3"  # Change to mistral if needed
OLLAMA_URL = "http://localhost:11434/api/generate"

def query_ollama_for_relations(text, drugs, diseases, organizations):
    """
    Query LLM API for relation extraction based on context + entities
    """
    prompt = f"""
You are an expert biomedical language model.

Given the text and extracted entities (Drugs, Diseases, Organizations),
identify CONTEXTUAL RELATIONS between them.

Use this format:
Entity1 ➝ relation ➝ Entity2

Only include relations that are clearly supported by the context.

Example Output:
Ponsegromab ➝ treats ➝ cancer cachexia
FDA ➝ approved ➝ ADLUMIZ
Biocon ➝ develops ➝ Kirsty

Text:
{text}

Entities:
Drugs: {', '.join(drugs) or 'None'}
Diseases: {', '.join(diseases) or 'None'}
Organizations: {', '.join(organizations) or 'None'}

Extracted Relations:
"""

    response = requests.post(OLLAMA_URL, json={
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    })

    if response.status_code == 200:
        output = response.json()["response"]
        return [line.strip() for line in output.split('\n') if '➝' in line]
    else:
        print(" ERROR from Ollama:", response.text)
        return []

def extract_relations_from_excel(input_path, flat_output_path, grouped_output_path):
    df = pd.read_excel(input_path)

    flat_records = []
    grouped_records = []

    for idx, row in df.iterrows():
        title = str(row.get("title", "")).strip()
        description = str(row.get("description", "")).strip()
        full_text = f"{title}\n{description}".strip()

        drugs = [x.strip() for x in str(row.get("drugs", "")).split(",") if x.strip()]
        diseases = [x.strip() for x in str(row.get("diseases", "")).split(",") if x.strip()]
        organizations = [x.strip() for x in str(row.get("organizations", "")).split(",") if x.strip()]
        countries = str(row.get("countries", "")).strip()

        if not any([drugs, diseases, organizations]):
            print(f"Row {idx+1} skipped (no entities).")
            continue

        print(f" Extracting from row {idx+1}/{len(df)}...")
        relations = query_ollama_for_relations(full_text, drugs, diseases, organizations)

        # Grouped output
        grouped_records.append({
            "Title": title,
            "Description": description,
            "Drugs": ', '.join(drugs),
            "Diseases": ', '.join(diseases),
            "Organizations": ', '.join(organizations),
            "Countries": countries,
            "Extracted Relations": '\n'.join(relations)
        })

        # Flat output
        for rel in relations:
            try:
                parts = [p.strip() for p in rel.split("➝")]
                if len(parts) == 3:
                    flat_records.append({
                        "Title": title,
                        "Description": description,
                        "Drugs": ', '.join(drugs),
                        "Diseases": ', '.join(diseases),
                        "Organizations": ', '.join(organizations),
                        "Countries": countries,
                        "Entity 1": parts[0],
                        "Relation": parts[1],
                        "Entity 2": parts[2]
                    })
            except:
                print(f" Failed to parse: {rel}")

    # Save results
    pd.DataFrame(grouped_records).to_excel(grouped_output_path, index=False)
    pd.DataFrame(flat_records).to_excel(flat_output_path, index=False)
    print(" Extraction complete.")


# ===  files paths  ===

input_excel = "C:/Users/EZB-VM-06/Downloads/_daily_news_070_gliner_largeModel_1028.xlsx"
output_flat = "C:/Users/EZB-VM-06/Desktop/News_codes_files/llama3/flat_relations_output_testing_news_3_daily_news.xlsx"
output_grouped = "C:/Users/EZB-VM-06/Desktop/News_codes_files/llama3/grouped_relations_output_testing_news_3_daily_news.xlsx"

# === Run ===
extract_relations_from_excel(input_excel, output_flat, output_grouped)
















# import requests

# # === CONFIGURATION ===
# OLLAMA_MODEL = "llama3"  # or use "mistral"
# OLLAMA_URL = "http://localhost:11434/api/generate"

# def query_ollama_for_relations(text, drugs, diseases, organizations):
#     """
#     Query LLM API for relation extraction based on context + entities
#     """
#     prompt = f"""
# You are an expert biomedical model.

# Given the following context and lists of extracted entities:
# Drugs: {', '.join(drugs) or 'None'}
# Diseases: {', '.join(diseases) or 'None'}
# Organizations: {', '.join(organizations) or 'None'}

# 🔍 Extract CONTEXTUAL RELATION TRIPLES formatted as:
# Entity1 ➝ relation ➝ Entity2

# Only include relations that are clearly implied by the context.
# Don't make up relations that have no context support.
# Do NOT include explanations. Just list the triples.

# Context:
# {text}

# Extracted Relations:
# """

#     response = requests.post(OLLAMA_URL, json={
#         "model": OLLAMA_MODEL,
#         "prompt": prompt,
#         "stream": False
#     })

#     if response.status_code == 200:
#         output = response.json()["response"]
#         return [line.strip() for line in output.split('\n') if '➝' in line]
#     else:
#         print("❌ Ollama error:", response.text)
#         return []

# # === 📝 INPUT: Long Paragraph and Entities ===

# title = "Lilly's orforglipron and other key catalysts set to reshape drug landscape"
# paragraph = """
# US pharma major Eli Lilly’s (NYSE: LLY) orforglipron is emerging as a promising oral glucagon-like peptide (GLP)-1 therapy, with upcoming clinical milestones in obesity and diabetes set to shape its market trajectory.
# As a wave of key catalysts - including US Food and Drug Administration (FDA) decisions and late-stage trial results - approaches in third-quarter 2025, the pharmaceutical landscape is poised for shifts driven by innovation, competitive positioning, and regulatory momentum, says pharma analytics company GlobalData.
# Lilly’s clinical data with orforglipron in patients with diabetes supports the drug’s potential as an obesity treatment, as per the Insights Investigative News team at GlobalData.
# This article is accessible to registered users, to continue reading please register for free. A free trial will give you access to exclusive features, interviews, round-ups and commentary from the sharpest minds in the pharmaceutical and biotechnology space for a week. If you are already a registered user please login. If your trial has come to an end, you can subscribe here.
# Try before you buy
# 7 day trial access
# Become a subscriber
# Or £77 per month
# The Pharma Letter is an extremely useful and valuable Life Sciences service that brings together a daily update on performance people and products. It’s part of the key information for keeping me informed
# Chairman, Sanofi Aventis UK
# Copyright © The Pharma Letter 2025 | Headless Content Management with Blaze
# """

# # Extracted entities (from GLiNER or manual):
# drugs = ["orforglipron"]
# diseases = ["diabetes", "obesity"]
# organizations = [
#     "Eli Lilly",
#     "GlobalData",
#     "Sanofi Aventis UK"

# ]

# # 🔁 Run relation extraction
# full_text = f"{title}\n{paragraph}"
# relations = query_ollama_for_relations(full_text, drugs, diseases, organizations)

# # ✅ Output the result
# print("\n✅ Extracted Biomedical Triples:\n")
# for i, rel in enumerate(relations, 1):
#     print(f"{i}. {rel}")