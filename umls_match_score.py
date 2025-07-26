import requests
import pandas as pd
import en_core_sci_sm
import time
from scispacy.linking import EntityLinker

# Load scispaCy model with UMLS linker
print("Loading scispaCy model...")
start_time = time.time()
nlp_umls = en_core_sci_sm.load()
nlp_umls.add_pipe("scispacy_linker", config={
    "resolve_abbreviations": True,
    "linker_name": "umls",
    "max_entities_per_mention": 1
})
linker_umls = nlp_umls.get_pipe("scispacy_linker")
print("Model loaded in", round(time.time() - start_time, 2), "seconds\n")

# UMLS API Key
UMLS_API_KEY = "b238480d-ef87-4755-a67c-92734e4dcfe8"

def get_umls_semantic_type(cui):
    """Get semantic type from UMLS API using CUI."""
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

# Store main results
main_results = []
# Store all candidates
all_candidates = []

# Terms to analyze
terms_to_check = ["AKT1", "akt1", "PKC-β", "PKCbeta","PKC-beta","protein kinase Cbeta","protein kinase C-β","protein kinase C beta","PKCbeta ( -/- )"]

for term in terms_to_check:
    try:
        candidates = linker_umls.candidate_generator([term], k=10)  # Get top 10 candidates
        if candidates and candidates[0]:
            top_candidate = max(candidates[0], key=lambda c: c.similarities[0])
            score = round(top_candidate.similarities[0], 4)
            alias_matched = top_candidate.aliases[0] if top_candidate.aliases else ""
            concept_id = top_candidate.concept_id
            umls_entity_data = linker_umls.kb.cui_to_entity[concept_id]

            main_results.append({
                "OriginalName": term,
                "UMLS_CUI": concept_id,
                "Matched_Alias": alias_matched,
                "Match_Score": score,
                "Canonical_Name": umls_entity_data.canonical_name,
                "UMLS_SemanticType": get_umls_semantic_type(concept_id)
            })

            # Save all candidates
            for cand in candidates[0]:
                umls_entity_data = linker_umls.kb.cui_to_entity[cand.concept_id]
                all_candidates.append({
                "OriginalName": term,
                "Candidate_CUI": cand.concept_id,
                "Canonical_Name": umls_entity_data.canonical_name,
                "Aliases": "; ".join(cand.aliases) if cand.aliases else "",
                "Match_Score": round(cand.similarities[0], 4)
            })


        else:
            main_results.append({
                "OriginalName": term,
                "UMLS_CUI": None,
                "Matched_Alias": "",
                "Match_Score": 0.0,
                "Canonical_Name": "",
                "UMLS_SemanticType": None
            })

    except Exception as e:
        print(f"Error processing term '{term}': {e}")

# Convert to DataFrames
df_main = pd.DataFrame(main_results)
df_candidates = pd.DataFrame(all_candidates)

# Print results
print("\n=== Main Top Match Results ===")
print(df_main)

print("\n=== All Candidates with Scores ===")
print(df_candidates)

# Save to CSV
df_main.to_csv("C:/Users/EZB-VM-06/Desktop/webapp_1/NERWebApp/RunNER/helper_functions/Nitin_files/UMLS_Top_Match_Results.csv", index=False)
df_candidates.to_csv("C:/Users/EZB-VM-06/Desktop/webapp_1/NERWebApp/RunNER/helper_functions/Nitin_files/UMLS_All_Candidates.csv", index=False)
print("\nResults saved to 'UMLS_Top_Match_Results.csv' and 'UMLS_All_Candidates.csv'")

## print the vectors for PKC-β to find how similarity score is working 
# doc = nlp_umls("PKC-β")
# print(doc.vector)  # This gives you a 300-dimensional vector for the term


