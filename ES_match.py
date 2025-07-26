import os
import pandas as pd

# === API gene data ===
genes = [
    {"orig_name": "C5aR", "name": "C5AR", "concept_id": "C1824180", "umls_conf_score": 0.9581},
    {"orig_name": "mice", "name": "MICE", "concept_id": "C1413131", "umls_conf_score": 0.9993},
    {"orig_name": "cell", "name": "CELL", "concept_id": "C1413336", "umls_conf_score": 0.9955},
    {"orig_name": "IgA", "name": "IGA", "concept_id": "C1413238", "umls_conf_score": 0.9976},
    {"orig_name": "WT mice", "name": "MICE", "concept_id": "C1413131", "umls_conf_score": 0.8159}
]


# === CONFIG ===
input_csv = "C:/Users/EZB-VM-06/Downloads/03_correction_applied.csv"
save_dir = "C:/Users/EZB-VM-06/Downloads"
os.makedirs(save_dir, exist_ok=True)
output_matched = os.path.join(save_dir, "07_gene_matched.csv")
output_unmatched = os.path.join(save_dir, "08_gene_unmatched.csv")

# === LOAD INPUT CSV ===
df = pd.read_csv(input_csv)

# Normalize for matching
df["Name_clean"] = df["Name"].astype(str).str.strip().str.lower()
df["FinalType_clean"] = df["FinalType"].astype(str).str.strip().str.lower()

# Filter only rows where FinalType is "Gene and Molecular Sequence"
gene_rows_df = df[df["FinalType_clean"] == "gene and molecular sequence"]

# Prepare gene names from API for exact match (case-insensitive)
api_gene_names = set(gene["orig_name"].strip().lower() for gene in genes)

# === MATCHED rows
matched_rows = []
for gene in genes:
    orig_name_clean = gene["orig_name"].strip().lower()
    matched_df = gene_rows_df[gene_rows_df["Name_clean"] == orig_name_clean]

    for _, row in matched_df.iterrows():
        matched_rows.append({
            "orig": gene["orig_name"],
            "matched": row["Name"],
            "match_score": gene["umls_conf_score"],
            "concept_id": gene["concept_id"]
        })

# === UNMATCHED rows from FinalType == "Gene and Molecular Sequence"
unmatched_gene_rows = gene_rows_df[~gene_rows_df["Name_clean"].isin(api_gene_names)]

# Drop helper columns before saving
unmatched_gene_rows = unmatched_gene_rows.drop(columns=["Name_clean", "FinalType_clean"])

# === SAVE OUTPUT
pd.DataFrame(matched_rows).to_csv(output_matched, index=False)
unmatched_gene_rows.to_csv(output_unmatched, index=False)

print(f" Matched rows: {len(matched_rows)}")
print(f" Unmatched 'Gene and Molecular Sequence' rows: {len(unmatched_gene_rows)}")
print(f" Saved matched to: {output_matched}")
print(f" Saved unmatched to: {output_unmatched}")
