# import pandas as pd
# import requests
# from difflib import SequenceMatcher

# # -------------------------------------------
# # SPECIAL_SYMBOLS dictionary (Excel file)
# SPECIAL_SYMBOLS = {
#     "PKCβ": "C1418913",
#     "PKC-β": "C1418913",
#     "protein kinase C (PKC)β": "C1418913",
#     "protein kinase C beta (PKCβ)": "C1418913",
#     "PKC-β2": "C1418913",
#     "PKCβ2": "C1418913",
#     "PKCβII": "C1418913",
#     "PKC-βII": "C1418913"
# }

# # -------------------------------------------
# # Function to get concept and class
# def get_umls_concept_and_umls_class_from_api(df):
#     concept_ids = []
#     matched_terms = []
#     umls_classes = []

#     for index, row in df.iterrows():
#         text = row['Name']

#         # First try to match from SPECIAL_SYMBOLS dictionary
#         matched_cui = None
#         for key, val in SPECIAL_SYMBOLS.items():
#             if text.lower() == key.lower():
#                 matched_cui = val
#                 concept_ids.append(val)
#                 matched_terms.append(key)
#                 umls_classes.append("Gene and Molecular Sequence")
#                 break

#         if matched_cui:
#             continue

#         # If no match, fallback to UMLS API
#         url = "https://uts-ws.nlm.nih.gov/rest/search/current"
#         params = {
#             "string": text,
#             "apiKey": "b238480d-ef87-4755-a67c-92734e4dcfe8",
#             "searchType": "exact"
#         }

#         try:
#             response = requests.get(url, params=params)
#             response.raise_for_status()
#             results = response.json()

#             result_items = results.get("result", {}).get("results", [])
#             if result_items:
#                 concept_id = result_items[0].get("ui", "")
#                 matched = result_items[0].get("name", "")
#                 semantic_type = result_items[0].get("rootSource", "Unknown")

#                 concept_ids.append(concept_id)
#                 matched_terms.append(matched)
#                 umls_classes.append(semantic_type)
#             else:
#                 concept_ids.append("")
#                 matched_terms.append("")
#                 umls_classes.append("")
#         except Exception as e:
#             print(f"API error for '{text}':", e)
#             concept_ids.append("")
#             matched_terms.append("")
#             umls_classes.append("")

#     df['concept_id'] = concept_ids
#     df['matched'] = matched_terms
#     df['umls_class'] = umls_classes
#     return df

# # -------------------------------------------
# # Function to correct abbreviations and assign final type
# def correct_abbreviations_data(df, abbr_dict, umls_class_to_type):
#     df['isAbbr'] = df['Name'].apply(lambda x: 'Yes' if x in abbr_dict else 'No')
#     df['OriginalName'] = df['Name']
#     df['Name'] = df.apply(lambda row: abbr_dict.get(row['Name'], row['Name']), axis=1)
#     df['FinalType'] = df['umls_class'].map(umls_class_to_type).fillna('OTHER')
#     return df, None

# # -------------------------------------------
# # Main Testing Script
# if __name__ == "__main__":
#     # Input term
#     test_gene = "PKC-β2"

#     # Create DataFrame
#     df = pd.DataFrame([{"Name": test_gene}])

#     print("🔍 Step 1: UMLS Lookup")
#     df = get_umls_concept_and_umls_class_from_api(df)

#     print("✅ Step 2: Correction & Tagging")
#     abbreviation_mapping = {
#         "PKC-β2": "PKCβ2"
#     }

#     umls_class_to_type = {
#         "Gene or Genome": "GENE",
#         "Gene and Molecular Sequence": "GENE",
#         "Amino Acid, Peptide, or Protein": "PROTEIN"
#     }

#     df, _ = correct_abbreviations_data(df, abbreviation_mapping, umls_class_to_type)

#     print("\n🧾 Final Output:")
#     print(df[['OriginalName', 'isAbbr', 'Name', 'concept_id', 'matched', 'umls_class', 'FinalType']])


# import pandas as pd
# import requests
# from difflib import SequenceMatcher
# import os

# # -------------------------------------------
# # Load SPECIAL_SYMBOLS from Excel file
# def load_special_symbols(excel_path):
#     """Load special symbols mapping from Excel file"""
#     try:
#         df = pd.read_excel(excel_path)
#         special_symbols = {}
#         for _, row in df.iterrows():
#             gene = str(row['Gene']).strip()
#             cui = str(row['Concept ID']).strip()
#             special_symbols[gene] = cui
#         print(f"✅ Loaded {len(special_symbols)} special symbols from Excel")
#         return special_symbols
#     except Exception as e:
#         print(f"❌ Error loading special symbols Excel: {e}")
#         return {}

# # Path to your Excel file
# SPECIAL_SYMBOLS_FILE = r"C:\Users\EZB-VM-06\Downloads\Gene_list_with_special_charecters_202507_Gene_with_special_charecter.xlsx"  # Update with your actual path
# SPECIAL_SYMBOLS = load_special_symbols(SPECIAL_SYMBOLS_FILE)

# # -------------------------------------------
# # Function to get concept and class
# def get_umls_concept_and_umls_class_from_api(df):
#     concept_ids = []
#     matched_terms = []
#     umls_classes = []

#     for index, row in df.iterrows():
#         text = row['Name']
#         text_lower = text.lower()

#         # First try to match from SPECIAL_SYMBOLS dictionary
#         matched_cui = None
#         matched_term = None
        
#         # Case-insensitive matching against special symbols
#         for key in SPECIAL_SYMBOLS.keys():
#             if text_lower == key.lower():
#                 matched_cui = SPECIAL_SYMBOLS[key]
#                 matched_term = key  # Preserve original case from mapping
#                 break

#         if matched_cui:
#             concept_ids.append(matched_cui)
#             matched_terms.append(matched_term)
#             umls_classes.append("Gene and Molecular Sequence")
#             continue

#         # If no match, fallback to UMLS API
#         url = "https://uts-ws.nlm.nih.gov/rest/search/current"
#         params = {
#             "string": text,
#             "apiKey": "b238480d-ef87-4755-a67c-92734e4dcfe8",
#             "searchType": "partial"
#         }

#         try:
#             response = requests.get(url, params=params)
#             response.raise_for_status()
#             results = response.json()

#             result_items = results.get("result", {}).get("results", [])
#             if result_items:
#                 concept_id = result_items[0].get("ui", "")
#                 matched = result_items[0].get("name", "")
#                 semantic_type = result_items[0].get("rootSource", "Unknown")

#                 concept_ids.append(concept_id)
#                 matched_terms.append(matched)
#                 umls_classes.append(semantic_type)
#             else:
#                 concept_ids.append("")
#                 matched_terms.append("")
#                 umls_classes.append("")
#         except Exception as e:
#             print(f"API error for '{text}':", e)
#             concept_ids.append("")
#             matched_terms.append("")
#             umls_classes.append("")

#     df['concept_id'] = concept_ids
#     df['matched'] = matched_terms
#     df['umls_class'] = umls_classes
#     return df

# # -------------------------------------------
# # Function to correct abbreviations and assign final type
# def correct_abbreviations_data(df, abbr_dict, umls_class_to_type):
#     df['isAbbr'] = df['Name'].apply(lambda x: 'Yes' if x in abbr_dict else 'No')
#     df['OriginalName'] = df['Name']
#     df['Name'] = df.apply(lambda row: abbr_dict.get(row['Name'], row['Name']), axis=1)
#     df['FinalType'] = df['umls_class'].map(umls_class_to_type).fillna('OTHER')
#     return df, None

# # -------------------------------------------
# # Main Testing Script
# if __name__ == "__main__":
#     # Input term
#     test_gene = "PKC-βII"

#     # Create DataFrame
#     df = pd.DataFrame([{"Name": test_gene}])

#     print("🔍 Step 1: UMLS Lookup")
#     df = get_umls_concept_and_umls_class_from_api(df)

#     print("✅ Step 2: Correction & Tagging")
#     abbreviation_mapping = {
#         "PKC-β2": "PKCβ2"
#     }

#     umls_class_to_type = {
#         "Gene or Genome": "GENE",
#         "Gene and Molecular Sequence": "GENE",
#         "Amino Acid, Peptide, or Protein": "PROTEIN"
#     }

#     df, _ = correct_abbreviations_data(df, abbreviation_mapping, umls_class_to_type)

#     print("\n🧾 Final Output:")
#     print(df[['OriginalName', 'isAbbr', 'Name', 'concept_id', 'matched', 'umls_class', 'FinalType']])




# import chardet
# import pandas as pd

# # Step 1: Detect encoding
# with open(r"C:\Users\EZB-VM-06\Downloads\Gene list with special charecters_2025-07 - Gene with special charecter.csv" , 'rb') as f:
#     result = chardet.detect(f.read(10000))  # Reads first 10KB to guess encoding

# print("Detected Encoding:", result)

# # Step 2: Read CSV with detected encoding
# df = pd.read_csv(r'C:\Users\EZB-VM-06\Downloads\Gene list with special charecters_2025-07 - Gene with special charecter.csv', encoding=result['encoding'])

######################################################################################
# import requests
# import pandas as pd
# import os
# import time
# import logging

# # Config
# SPECIAL_SYMBOLS_FILE = "C:/Users/EZB-VM-06/Downloads/Gene list with special charecters_2025-07 - Gene with special charecter.csv"
# SEM_TYPE_FILEPATH = "C:/Users/EZB-VM-06/Downloads/MRSTY.RRF"
# OUTPUT_LOG_CSV = "C:/Users/EZB-VM-06/Desktop/webapp_1/NERWebApp/RunNER/helper_functions/Nitin_files/processed_single_term.csv"
# API_KEY = "b238480d-ef87-4755-a67c-92734e4dcfe8"
# BASE_URI = 'https://uts-ws.nlm.nih.gov'
# LOG_FILE_PATH = "single_term_lookup.log"

# # Logging setup
# log_dir = os.path.dirname(LOG_FILE_PATH)
# if log_dir:
#     os.makedirs(log_dir, exist_ok=True)

# logging.basicConfig(filename=LOG_FILE_PATH, filemode='a', format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Globals
# SPECIAL_SYMBOLS_MAP = {}
# CUI_SEMANTIC_TYPES = {}

# def initialize_resources():
#     global SPECIAL_SYMBOLS_MAP, CUI_SEMANTIC_TYPES

#     # Load special symbols
#     encodings = ['utf-8', 'utf-8-sig', 'latin1', 'cp1252']
#     for enc in encodings:
#         try:
#             df = pd.read_csv(SPECIAL_SYMBOLS_FILE, encoding=enc)
#             print(f"Loaded {len(df)} special symbol mappings using encoding: {enc}")
#             break
#         except Exception as e:
#             print(f"Failed with encoding {enc}: {e}")
#     else:
#         print("Failed to load special symbols file")
#         return

#     for _, row in df.iterrows():
#         gene = row['Gene'].strip()
#         cui = row['Concept ID'].strip()
#         SPECIAL_SYMBOLS_MAP[gene] = cui

#     # Load semantic types
#     with open(SEM_TYPE_FILEPATH, 'r', encoding='utf-8') as f:
#         for line in f:
#             fields = line.strip().split('|')
#             cui, tui, stn, sty = fields[0], fields[1], fields[2], fields[3]
#             if cui not in CUI_SEMANTIC_TYPES:
#                 CUI_SEMANTIC_TYPES[cui] = []
#             CUI_SEMANTIC_TYPES[cui].append({'TUI': tui, 'STY': sty, 'STN': stn})

# def get_umls_cui(text):
#     if text is None:
#         logger.warning("get_umls_cui received None")
#         return None, None

#     if text in SPECIAL_SYMBOLS_MAP:
#         cui = SPECIAL_SYMBOLS_MAP[text]
#         logger.info(f"Found in SPECIAL_SYMBOLS_MAP: {text} -> {cui}")
#         return cui, {
#             'result': {
#                 'results': [{
#                     'ui': cui,
#                     'name': text
#                 }]
#             }
#         }

#     # API fallback
#     url = f"{BASE_URI}/rest/search/current/?string={text}&partialSearch=true"
#     try:
#         response = requests.get(url, params={"apiKey": API_KEY})
#         if response.status_code == 200:
#             data = response.json()
#             cui = data['result']['results'][0]['ui']
#             return cui, data
#         else:
#             logger.error(f"UMLS API failed: {response.status_code}")
#             return None, None
#     except Exception as e:
#         logger.exception(f"Exception in get_umls_cui for '{text}': {e}")
#         return None, None

# def get_sem_type_for_cui(cui):
#     types = CUI_SEMANTIC_TYPES.get(cui)
#     if types:
#         return types[0]['STY']
#     return None

# def get_umls_api_data(cui):
#     url = f"{BASE_URI}/rest/content/current/CUI/{cui}"
#     try:
#         response = requests.get(url, params={"apiKey": API_KEY})
#         if response.status_code == 200:
#             result = response.json().get('result', {})
#             sty_list = result.get('semanticTypes', [])
#             if sty_list:
#                 return sty_list[0].get('name'), result
#         return None, None
#     except Exception as e:
#         logger.exception(f"Exception in get_umls_api_data for CUI '{cui}': {e}")
#         return None, None

# def get_umls_concept_and_umls_class_from_api(entity_df):
#     logger.info("Starting UMLS concept processing...")
#     processed_terms = []
#     cui_map = {}

#     for string in entity_df['Name'].unique():
#         cui = None
#         name = None
#         source = "NO_MATCH"

#         if string in SPECIAL_SYMBOLS_MAP:
#             cui = SPECIAL_SYMBOLS_MAP[string]
#             name = string
#             source = "SPECIAL_SYMBOLS_MAP"
#             logger.info(f"Matched '{string}' via SPECIAL_SYMBOLS_MAP -> CUI: {cui}")
#         else:
#             query = {'string': string, 'apiKey': API_KEY, 'pageNumber': 1, 'partialSearch': 'true'}
#             try:
#                 response = requests.get(BASE_URI + "/search/current", params=query, timeout=10)
#                 if response.status_code == 200:
#                     results = response.json().get('result', {}).get('results', [])
#                     if results:
#                         cui = results[0]['ui']
#                         name = results[0]['name']
#                         source = "API"
#                 else:
#                     logger.error(f"API error for {string}: {response.status_code}")
#             except Exception as e:
#                 logger.exception(f"Exception for term '{string}': {e}")

#         cui_map[string] = {"concept_id": cui, "name": name}
#         processed_terms.append({
#             "term": string,
#             "match": "YES" if cui else "NO",
#             "cui": cui or "",
#             "source": source
#         })

#     df_processed = pd.DataFrame(processed_terms)
#     df_processed.to_csv(OUTPUT_LOG_CSV, index=False)
#     logger.info("Saved processed_terms to CSV.")

#     umls_entities = []
#     for _, row in entity_df.iterrows():
#         ent = row['Name']
#         cui_data = cui_map.get(ent, {})
#         cui = cui_data.get("concept_id")
#         name = cui_data.get("name")

#         if cui:
#             umls_entities.append({
#                 "match_flag": 1,
#                 "orig": ent,
#                 "matched": name,
#                 "match_score": 1.0,
#                 "concept_id": cui,
#                 "canonical_name": None,
#                 "defination": None,
#                 "aliases": None,
#                 "source": source
#             })
#         else:
#             umls_entities.append({
#                 "match_flag": 0,
#                 "orig": ent,
#                 "matched": None,
#                 "match_score": 0.0,
#                 "concept_id": None,
#                 "canonical_name": None,
#                 "defination": None,
#                 "aliases": None,
#                 "source": source
#             })

#     umls_df = pd.DataFrame(umls_entities)
#     cui_class_map = {}
#     for cui in umls_df['concept_id'].dropna().unique():
#         sem = get_sem_type_for_cui(cui)
#         if not sem:
#             sem, _ = get_umls_api_data(cui)
#         cui_class_map[cui] = sem

#     umls_df['umls_class'] = umls_df['concept_id'].apply(lambda x: cui_class_map.get(x))
#     final_df = pd.merge(entity_df, umls_df, left_index=True, right_index=True)
#     return final_df

# if __name__ == "__main__":
#     initialize_resources()
#     # input_term = input("Enter term to look up: ").strip()
#     # df = pd.DataFrame([{"Name": input_term}])
#     # result_df = get_umls_concept_and_umls_class_from_api(df)

#     # Example: Load your entity list from a CSV file (replace path as needed)
#     ENTITY_INPUT_PATH = "C:/Users/EZB-VM-06/Downloads/entity_df.xlsx"  # This file must have a 'Name' column
#     entity_df = pd.read_excel(ENTITY_INPUT_PATH)

#     # Optional: Clean 'Name' column
#     entity_df['Name'] = entity_df['Name'].astype(str).str.strip()

#     # Run UMLS concept matching
#     result_df = get_umls_concept_and_umls_class_from_api(entity_df)


#     print("\n=== UMLS Lookup Result ===")
#     print(result_df[['Name', 'matched', 'concept_id', 'umls_class', 'source', 'match_flag']].to_string(index=False))




import pandas as pd
import re

def search_genes(csv_file, search_terms):
    """
    Robust gene search function that handles common issues
    """
    # Read CSV with UTF-8 encoding
    df = pd.read_csv(csv_file, encoding='utf-8')
    
    # Strip whitespace from all string columns
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    
    # Print first few rows to verify data
    print("First 5 rows of data:")
    print(df.head())
    print(f"\nTotal rows: {len(df)}")
    
    results = {}
    
    for term in search_terms:
        print(f"\n--- Searching for: '{term}' ---")
        
        # Method 1: Exact match
        exact_match = df[df['Gene'] == term]
        
        # Method 2: Case-insensitive contains
        contains_match = df[df['Gene'].str.contains(term, case=False, na=False)]
        
        # Method 3: Regex search (escape special characters)
        regex_term = re.escape(term)
        regex_match = df[df['Gene'].str.contains(regex_term, regex=True, na=False)]
        
        # Store results
        results[term] = {
            'exact': exact_match,
            'contains': contains_match,
            'regex': regex_match
        }
        
        # Print results
        print(f"Exact matches: {len(exact_match)}")
        print(f"Contains matches: {len(contains_match)}")
        print(f"Regex matches: {len(regex_match)}")
        
        if len(exact_match) > 0:
            print("Exact match results:")
            print(exact_match[['Concept ID', 'Gene']])
        elif len(contains_match) > 0:
            print("Contains match results:")
            print(contains_match[['Concept ID', 'Gene']])
        elif len(regex_match) > 0:
            print("Regex match results:")
            print(regex_match[['Concept ID', 'Gene']])
        else:
            print("No matches found")
            
            # Debug: Show similar entries
            print("Similar entries (first 10):")
            similar = df[df['Gene'].str.contains('PKC', case=False, na=False)].head(10)
            print(similar[['Concept ID', 'Gene']])
    
    return results

# Example usage
if __name__ == "__main__":
    # Your search terms
    search_terms = ['PKC-β2', 'PKCβII', 'PKCβI', 'PKCβ','PKCδ']
    
    # Perform search
    results = search_genes('C:/Users/EZB-VM-06/Downloads/Gene list with special charecters_2025-07 - Gene with special charecter.csv', search_terms)
    
    # Alternative: Simple search function
    def simple_search(df, term):
        # Clean the data
        df_clean = df.copy()
        df_clean['Gene'] = df_clean['Gene'].str.strip()
        
        # Search
        return df_clean[df_clean['Gene'] == term]
    
    # If you just want a quick solution:
    df = pd.read_csv('C:/Users/EZB-VM-06/Downloads/Gene list with special charecters_2025-07 - Gene with special charecter.csv', encoding='utf-8')
    df['Gene'] = df['Gene'].str.strip()  # Remove whitespace
    
    # Search for each term
    for term in search_terms:
        result = df[df['Gene'] == term]
        print(f"\nSearching for '{term}':")
        print(f"Found {len(result)} matches")
        if len(result) > 0:
            print(result)