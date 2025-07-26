# import pandas as pd

# # Load the first CSV file with specified columns
# file1_columns = ['Entity1','E1_Concept_ID']

# df1 = pd.read_csv('C:/Users/EZB-VM-06/Desktop/Database_comparison/Model_files/39287853/after_entity_correction.csv', usecols=file1_columns)

# # Load the second CSV file with specified columns
# file2_columns = ['entity1', 'e1_concept_id']
# df2 = pd.read_csv('C:/Users/EZB-VM-06/Desktop/Database_comparison/Database_files/39287853/relation_extraction_outputs_new_202507011700.csv', usecols=file2_columns)

# # Clean and standardize the concept ID columns (in case they're different data types)
# df1['E1_Concept_ID'] = df1['E1_Concept_ID'].astype(str).str.strip()
# df2['e1_concept_id'] = df2['e1_concept_id'].astype(str).str.strip()

# # Find matches - rows in df1 where E1_Concept_ID exists in df2's e1_concept_id
# matched_rows = df1[df1['E1_Concept_ID'].isin(df2['e1_concept_id'])]

# # Find unmatched rows - rows in df1 where E1_Concept_ID doesn't exist in df2
# unmatched_rows = df1[~df1['E1_Concept_ID'].isin(df2['e1_concept_id'])]

# # Save the results to new CSV files
# matched_rows.to_csv('C:/Users/EZB-VM-06/Desktop/Database_comparison/Comparison_files/39287853/matched_records.csv', index=False)
# unmatched_rows.to_csv('C:/Users/EZB-VM-06/Desktop/Database_comparison/Comparison_files/39287853/unmatched_records.csv', index=False)

# print("Processing complete!")
# print(f"Matched records saved to 'matched_records.csv' ({len(matched_rows)} rows)")
# print(f"Unmatched records saved to 'unmatched_records.csv' ({len(unmatched_rows)} rows)")





import requests
import pandas as pd
import en_core_sci_sm
import time
from scispacy.linking import EntityLinker

# Load scispaCy model with UMLS linker (same config as mydifflib.py)
print("Loading scispaCy model...")
start_time = time.time()
nlp_umls = en_core_sci_sm.load()
nlp_umls.add_pipe("scispacy_linker", config={
    "resolve_abbreviations": True,
    "linker_name": "umls",
    "max_entities_per_mention": 1  # Same as mydifflib.py
})
linker_umls = nlp_umls.get_pipe("scispacy_linker")
print("Model loaded in", round(time.time() - start_time, 2), "seconds\n")

# UMLS API Key (same as mydifflib.py)
UMLS_API_KEY = "b238480d-ef87-4755-a67c-92734e4dcfe8"

def get_umls_semantic_type(cui):
    """Same as get_sem_type_for_cui() in mydifflib.py"""
    if cui:
        url = f"https://uts-ws.nlm.nih.gov/rest/content/current/CUI/{cui}"
        params = {"apiKey": UMLS_API_KEY}
        try:
            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                types = data['result'].get('semanticTypes', [])
                if types:
                    return types[0]['name']
        except Exception as e:
            print(f"Error fetching semantic type for CUI {cui}: {e}")
    return None

# Store results (same format as mydifflib.py)
results = []

# Terms to analyze
terms_to_check = ["AKT1", "akt1", "PKC-β", "PKCbeta", "PKC-beta", 
                 "protein kinase Cbeta", "protein kinase C-β", 
                 "protein kinase C beta", "PKCbeta ( -/- )"]

for term in terms_to_check:
    try:
        # EXACT SAME LOGIC AS mydifflib.py:
        # 1. Get candidates (k=1, since max_entities_per_mention=1)
        candidates = linker_umls.candidate_generator([term], k=1)
        
        if candidates and candidates[0] and len(candidates[0]) > 0:
            # 2. Take the first candidate (no max selection, since k=1)
            candidate = candidates[0][0]
            score = candidate.similarities[0]
            alias_matched = candidate.aliases[0] if candidate.aliases else term
            concept_id = candidate.concept_id
            
            # 3. Get UMLS entity data (same as mydifflib.py)
            umls_entity_data = linker_umls.kb.cui_to_entity[concept_id]
            
            # 4. Append results (same format)
            results.append({
                "OriginalName": term,
                "UMLS_CUI": concept_id,
                "Matched_Alias": alias_matched,
                "Match_Score": score,  # Same raw score (no rounding)
                "Canonical_Name": umls_entity_data.canonical_name,
                "UMLS_SemanticType": get_umls_semantic_type(concept_id)
            })
        else:
            # No match (same as mydifflib.py)
            results.append({
                "OriginalName": term,
                "UMLS_CUI": None,
                "Matched_Alias": term,  # Original term if no match
                "Match_Score": 0.0,
                "Canonical_Name": "",
                "UMLS_SemanticType": None
            })
    except Exception as e:
        print(f"Error processing term '{term}': {e}")

# Convert to DataFrame (same as mydifflib.py)
df_results = pd.DataFrame(results)

# Print results
print("\n=== Results (Same Logic as mydifflib.py) ===")
print(df_results)

# Save to CSV
df_results.to_csv("C:/Users/EZB-VM-06/Desktop/webapp_1/NERWebApp/RunNER/helper_functions/Nitin_files/UMLS_Match_Results_Same_As_mydifflib.csv", index=False)
print("\nResults saved to 'UMLS_Match_Results_Same_As_mydifflib.csv'")