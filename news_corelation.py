



import pandas as pd
import re

def extract_entity_relations(text, drugs, diseases, organizations):
    """
    Extract Drug-Disease-Organization relations from a biomedical text.
    """
    text_lower = text.lower()
    sentences = re.split(r'(?<=[.!?])\s+', text)
    relations = []

    # Keywords
    approval_patterns = ["approved", "approval", "cleared by"]
    announcement_patterns = ["announced", "declared", "stated", "shared"]
    treat_patterns = ["treat", "treatment", "used for", "used to treat", "indicated to", "indicated to treat", "improve glycemic control"]

    # Normalize
    drugs_lower = [d.lower() for d in drugs]
    diseases_lower = [d.lower() for d in diseases]
    orgs_lower = [o.lower() for o in organizations]

    for sentence in sentences:
        sent = sentence.strip()
        sent_lower = sent.lower()

        for drug in drugs:
            if drug and drug.lower() in sent_lower:
                for disease in diseases:
                    if disease and disease.lower() in sent_lower:
                        if any(p in sent_lower for p in treat_patterns):
                            relations.append(f"{drug} ➝ treats ➝ {disease}")

        for org in organizations:
            if org and org.lower() in sent_lower:
                for drug in drugs:
                    if drug and drug.lower() in sent_lower:
                        if any(p in sent_lower for p in approval_patterns):
                            relations.append(f"{org} ➝ approved ➝ {drug}")

        for org in organizations:
            if org and org.lower() in sent_lower:
                if any(p in sent_lower for p in announcement_patterns):
                    for drug in drugs:
                        if drug and drug.lower() in sent_lower:
                            if any(p in sent_lower for p in approval_patterns):
                                relations.append(f"{org} ➝ announced_approval_of ➝ {drug}")

    return list(set(relations))


def process_csv_grouped_per_row(input_excel_path, output_excel_path):
    df = pd.read_excel(input_excel_path)
    output_rows = []

    for idx, row in df.iterrows():
        title = str(row.get("title", "")).strip()
        description = str(row.get("description", "")).strip()
        country = str(row.get("countries", "")).strip()

        text = f"{title} {description}".strip()
        drugs = [x.strip() for x in str(row.get("drugs", "")).split(",") if x.strip()]
        diseases = [x.strip() for x in str(row.get("diseases", "")).split(",") if x.strip()]
        organizations = [x.strip() for x in str(row.get("organizations", "")).split(",") if x.strip()]

        if not any([drugs, diseases, organizations]):
            continue

        relations = extract_entity_relations(text, drugs, diseases, organizations)

        output_rows.append({
            "Title": title,
            "Description": description,
            "Drugs": ", ".join(drugs),
            "Diseases": ", ".join(diseases),
            "Organizations": ", ".join(organizations),
            "Countries": country,
            "Extracted Relations": "\n".join(relations)
        })

        print(f"✅ Processed row {idx + 1}/{len(df)}")

    if output_rows:
        pd.DataFrame(output_rows).to_excel(output_excel_path, index=False)
        print(f"\n📁 Saved grouped results to: {output_excel_path}")
    else:
        print("⚠ No relations extracted.")


# === Run === #

input_excel_path = "C:/Users/EZB-VM-06/Downloads/070_gliner_largeModel_1028 (1).xlsx"
output_excel_path = "C:/Users/EZB-VM-06/Downloads/grouped_extract_relations.xlsx"

process_csv_grouped_per_row(input_excel_path, output_excel_path)

################# below code is not extracting relations properly but we can used after ####################

# import pandas as pd
# import re

# # Define biomedical entity types (optional for future role tagging extensions)
# BIOMED_ENTITY_TYPES = [
#     "Drug", "Antigen", "Gene", "Variant", "GeneFamily",
#     "Disease", "Pathway", "Therapy", "Biomarker"
# ]

# # Define relation phrases for logical triples
# RELATION_TYPES = {
#     "treats": ["treat", "treated", "used for", "indicated in", "used to treat", "used in treatment"],
#     "targets": ["targets", "binds to", "modulates", "acts on", "inhibits"],
#     "associated_with": ["associated with", "linked to", "correlated with"],
#     "causes": ["causes", "leads to", "results in", "is responsible for"],
#     "variant_of": ["variant of", "mutation of", "isoform of"],
#     "developed_by": ["developed by", "engineered by", "created by"],
#     "announced_approval_of": ["announced.*approval", "stated approval of"],
#     "approved": ["approved", "approval", "cleared by"]
# }


# def extract_triples(text, entities):
#     """
#     Extract biomedical relation triples from free text using heuristics.
#     """
#     sentences = re.split(r'(?<=[.!?])\s+', text.lower())
#     triples = []

#     # Create a normalized dictionary {lowercase entity: original text}
#     norm_entities = {e.lower(): e for e in entities}

#     # Create pairwise entity combinations
#     for sent in sentences:
#         for ent1_raw in entities:
#             ent1 = ent1_raw.lower()
#             if ent1 not in sent:
#                 continue
#             for ent2_raw in entities:
#                 ent2 = ent2_raw.lower()
#                 if ent1 == ent2 or ent2 not in sent:
#                     continue

#                 for rel, patterns in RELATION_TYPES.items():
#                     for pattern in patterns:
#                         regex = rf"{re.escape(ent1)}.*{pattern}.*{re.escape(ent2)}"
#                         if re.search(regex, sent):
#                             triples.append((norm_entities[ent1], rel, norm_entities[ent2]))
#     return list(set(triples))  # Remove duplicates


# def process_articles_with_roles(input_path, grouped_output_path, flat_output_path):
#     df = pd.read_excel(input_path)
#     grouped_rows = []
#     flat_rows = []

#     for idx, row in df.iterrows():
#         title = str(row.get("title", "")).strip()
#         description = str(row.get("description", "")).strip()
#         country = str(row.get("countries", "")).strip()

#         drugs = [x.strip() for x in str(row.get("drugs", "")).split(",") if x.strip()]
#         diseases = [x.strip() for x in str(row.get("diseases", "")).split(",") if x.strip()]
#         organizations = [x.strip() for x in str(row.get("organizations", "")).split(",") if x.strip()]

#         full_text = f"{title} {description}"
#         all_entities = drugs + diseases + organizations

#         if not all_entities:
#             print(f"⚠ Skipped row {idx+1}: No entities found.")
#             continue

#         triples = extract_triples(full_text, all_entities)

#         # Grouped Output
#         grouped_rows.append({
#             "Title": title,
#             "Description": description,
#             "Drugs": ", ".join(drugs),
#             "Diseases": ", ".join(diseases),
#             "Organizations": ", ".join(organizations),
#             "Countries": country,
#             "Extracted Relations": "\n".join(f"{s} ➝ {r} ➝ {o}" for s, r, o in triples)
#         })

#         # Flat Output
#         for s, r, o in triples:
#             flat_rows.append({
#                 "Title": title,
#                 "Description": description,
#                 "Drugs": ", ".join(drugs),
#                 "Diseases": ", ".join(diseases),
#                 "Organizations": ", ".join(organizations),
#                 "Countries": country,
#                 "Entity 1": s,
#                 "Relation": r,
#                 "Entity 2": o
#             })

#         print(f"✅ Processed row {idx+1}/{len(df)}")

#     # Save Grouped Output
#     if grouped_rows:
#         grouped_df = pd.DataFrame(grouped_rows)
#         grouped_df.to_excel(grouped_output_path, index=False)
#         print(f"📁 Grouped output saved to: {grouped_output_path}")

#     # Save Flat Output
#     if flat_rows:
#         flat_df = pd.DataFrame(flat_rows)
#         flat_df.to_excel(flat_output_path, index=False)
#         print(f"📁 Flat output saved to: {flat_output_path}")


# # === ✅ RUN THE PIPELINE ===

# input_excel_path = "C:/Users/EZB-VM-06/Downloads/070_gliner_largeModel_1028 (1).xlsx"
# grouped_output_path = "C:/Users/EZB-VM-06/Downloads/grouped_triples_output.xlsx"
# flat_output_path = "C:/Users/EZB-VM-06/Downloads/flattened_triples_output.xlsx"

# process_articles_with_roles(input_excel_path, grouped_output_path, flat_output_path)