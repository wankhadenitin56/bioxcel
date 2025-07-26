# #!/usr/bin/env python3


# import pandas as pd
# import numpy as np
# import requests
# import json
# import time
# from typing import Dict, List, Optional, Tuple

# try:
#     import spacy
#     from scispacy.linking import EntityLinker
#     from scispacy.abbreviation import AbbreviationDetector
# except ImportError:
#     print("Required packages not installed. Please install:")
#     print("pip install scispacy")
#     print("pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/en_core_sci_sm-0.5.3.tar.gz")
#     exit(1)

# class UMLSConfidenceCalculator:
#     """
#     Complete UMLS Confidence Score Calculator
#     """
    
#     def __init__(self, model_name="en_core_sci_sm", linker_name="umls"):
#         """
#         Initialize the UMLS confidence calculator
        
#         Args:
#             model_name (str): Spacy model name (en_core_sci_sm or en_core_sci_md)
#             linker_name (str): Entity linker name (umls, mesh, rxnorm, go, hpo)
#         """
#         self.model_name = model_name
#         self.linker_name = linker_name
#         self.nlp = None
#         self.linker_umls = None
#         self.cui_class_cache = {}
        
#         # UMLS API credentials 
#         self.umls_api_key = None  
        
#         self._initialize_models()
    
#     def _initialize_models(self):
#         """Initialize spaCy model and UMLS linker"""
#         try:
#             print(f"Loading spaCy model: {self.model_name}")
#             self.nlp = spacy.load(self.model_name)
            
#             print(f"Loading UMLS linker: {self.linker_name}")
#             self.linker_umls = EntityLinker(resolve_abbreviations=True, name=self.linker_name)
#             self.nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": self.linker_name})
            
#             print("Models loaded successfully!")
            
#         except Exception as e:
#             print(f"Error loading models: {str(e)}")
#             print("Make sure you have installed the required packages and models")
#             raise
    
#     def get_umls_api_data(self, cui: str) -> Tuple[str, str]:
        
#         if not self.umls_api_key:
#             return ("Unknown", "API key not provided")
        
#         try:
#             # UMLS API endpoint
#             url = f"https://uts-ws.nlm.nih.gov/rest/content/current/CUI/{cui}"
#             params = {
#                 'apiKey': self.umls_api_key,
#                 'returnIdType': 'concept'
#             }
            
#             response = requests.get(url, params=params)
#             if response.status_code == 200:
#                 data = response.json()
#                 semantic_type = data.get('result', {}).get('semanticTypes', [{}])[0].get('name', 'Unknown')
#                 definition = data.get('result', {}).get('definition', 'No definition available')
#                 return (semantic_type, definition)
#             else:
#                 return ("API Error", "Could not retrieve data")
                
#         except Exception as e:
#             return ("Error", str(e))
    
#     def get_sem_type_for_cui(self, cui: str) -> Optional[str]:
        
#         try:
#             if cui in self.linker_umls.kb.cui_to_entity:
#                 entity = self.linker_umls.kb.cui_to_entity[cui]
#                 # Try to get semantic types from entity
#                 if hasattr(entity, 'types') and entity.types:
#                     return entity.types[0] if entity.types else None
#             return None
#         except:
#             return None
    
#     def calculate_confidence_score(self, word1: str, word2: str) -> Dict:
        
#         results = {}
#         words = [word1, word2]
        
#         print(f"Calculating confidence scores for: '{word1}' and '{word2}'")
#         print("-" * 50)
        
#         for idx, word in enumerate(words, 1):
#             word_key = f"word_{idx}"
            
#             # Initialize result structure
#             results[word_key] = {
#                 "original_word": word,
#                 "match_flag": 0,
#                 "matched_alias": "",
#                 "confidence_score": 0.0,
#                 "concept_id": "",
#                 "canonical_name": "",
#                 "definition": "",
#                 "aliases": [],
#                 "umls_class": ""
#             }
            
#             if word is None or word.strip() == "":
#                 print(f"Word {idx}: Empty or None - skipping")
#                 continue
            
#             try:
#                 print(f"Processing word {idx}: '{word}'")
                
#                 # Get UMLS candidates using the same logic as original code
#                 candidates = self.linker_umls.candidate_generator([word], k=1)
                
#                 if len(candidates) > 0 and len(candidates[0]) > 0:
#                     # Extract best candidate
#                     best_candidate = candidates[0][0]
                    
#                     # Get confidence score (similarity score)
#                     confidence_score = best_candidate.similarities[0]
#                     matched_alias = best_candidate.aliases[0]
#                     concept_id = best_candidate.concept_id
                    
#                     # Get full UMLS entity data
#                     umls_entity_data = self.linker_umls.kb.cui_to_entity[concept_id]
                    
#                     # Get UMLS class/semantic type
#                     umls_class = self.get_sem_type_for_cui(concept_id)
#                     if umls_class is None:
#                         umls_class = self.get_umls_api_data(concept_id)[0]
                    
#                     # Update results
#                     results[word_key].update({
#                         "match_flag": 1,
#                         "matched_alias": matched_alias,
#                         "confidence_score": confidence_score,
#                         "concept_id": umls_entity_data.concept_id,
#                         "canonical_name": umls_entity_data.canonical_name,
#                         "definition": umls_entity_data.definition,
#                         "aliases": umls_entity_data.aliases,
#                         "umls_class": umls_class
#                     })
                    
#                     print(f"  ✓ Match found - Confidence: {confidence_score:.4f}")
                    
#                 else:
#                     print(f"  ✗ No UMLS match found")
                    
#             except Exception as e:
#                 print(f"  ✗ Error processing word '{word}': {str(e)}")
        
#         return results
    
#     def print_detailed_results(self, results: Dict):
#         """Print detailed confidence score results"""
        
#         print("\n" + "=" * 70)
#         print("DETAILED UMLS CONFIDENCE SCORE RESULTS")
#         print("=" * 70)
        
#         for word_key, data in results.items():
#             print(f"\n{word_key.upper().replace('_', ' ')}:")
#             print(f"  Original Word: {data['original_word']}")
#             print(f"  Match Found: {'Yes' if data['match_flag'] else 'No'}")
            
#             if data['match_flag']:
#                 print(f"  Confidence Score: {data['confidence_score']:.4f}")
#                 print(f"  Matched Alias: {data['matched_alias']}")
#                 print(f"  Concept ID: {data['concept_id']}")
#                 print(f"  Canonical Name: {data['canonical_name']}")
#                 print(f"  UMLS Class: {data['umls_class']}")
                
#                 # Truncate long definitions
#                 definition = data['definition']
#                 if len(definition) > 100:
#                     definition = definition[:100] + "..."
#                 print(f"  Definition: {definition}")
                
#                 # Show first few aliases
#                 aliases = data['aliases']
#                 if len(aliases) > 3:
#                     aliases_str = ", ".join(aliases[:3]) + f" (and {len(aliases)-3} more)"
#                 else:
#                     aliases_str = ", ".join(aliases)
#                 print(f"  Aliases: {aliases_str}")
#             else:
#                 print(f"  Confidence Score: {data['confidence_score']}")
#                 print("  No UMLS match found")
    
#     def compare_confidence_scores(self, results: Dict):
#         """Compare confidence scores between two words"""
        
#         word1_data = results['word_1']
#         word2_data = results['word_2']
        
#         word1_score = word1_data['confidence_score']
#         word2_score = word2_data['confidence_score']
        
#         print(f"\n" + "=" * 50)
#         print("CONFIDENCE SCORE COMPARISON")
#         print("=" * 50)
#         print(f"Word 1 ({word1_data['original_word']}): {word1_score:.4f}")
#         print(f"Word 2 ({word2_data['original_word']}): {word2_score:.4f}")
        
#         difference = abs(word1_score - word2_score)
        
#         if word1_score > word2_score:
#             print(f"→ Word 1 has higher confidence (+{difference:.4f})")
#             print(f"→ Word 1 is more confidently matched to UMLS")
#         elif word2_score > word1_score:
#             print(f"→ Word 2 has higher confidence (+{difference:.4f})")
#             print(f"→ Word 2 is more confidently matched to UMLS")
#         else:
#             print("→ Both words have equal confidence scores")
        
#         # Interpretation
#         print(f"\nINTERPRETATION:")
#         for i, (word_key, data) in enumerate(results.items(), 1):
#             score = data['confidence_score']
#             word = data['original_word']
            
#             if score >= 0.9:
#                 confidence_level = "Very High"
#             elif score >= 0.7:
#                 confidence_level = "High"
#             elif score >= 0.5:
#                 confidence_level = "Moderate"
#             elif score >= 0.3:
#                 confidence_level = "Low"
#             else:
#                 confidence_level = "Very Low"
            
#             print(f"  Word {i} ('{word}'): {confidence_level} confidence ({score:.4f})")
    
#     def export_results_to_dataframe(self, results: Dict) -> pd.DataFrame:
#         """Export results to pandas DataFrame"""
        
#         rows = []
#         for word_key, data in results.items():
#             rows.append({
#                 'word_number': word_key,
#                 'original_word': data['original_word'],
#                 'match_flag': data['match_flag'],
#                 'confidence_score': data['confidence_score'],
#                 'matched_alias': data['matched_alias'],
#                 'concept_id': data['concept_id'],
#                 'canonical_name': data['canonical_name'],
#                 'umls_class': data['umls_class'],
#                 'definition': data['definition']
#             })
        
#         return pd.DataFrame(rows)

# def main():
#     """Main function to demonstrate usage"""
    
#     print("UMLS Confidence Score Calculator")
#     print("=" * 40)
    
#     # Initialize calculator
#     try:
#         calculator = UMLSConfidenceCalculator()
#     except Exception as e:
#         print(f"Failed to initialize calculator: {e}")
#         return
    
#     # Example words (you can change these)
#     word1 = "PKC-β"
#     word2 = "AKT1"
    
#     # You can also get input from user
#     # word1 = input("Enter first word: ").strip()
#     # word2 = input("Enter second word: ").strip()
    
#     # Calculate confidence scores
#     results = calculator.calculate_confidence_score(word1, word2)
    
#     # Display results
#     calculator.print_detailed_results(results)
#     calculator.compare_confidence_scores(results)
    
#     # Export to DataFrame
#     df = calculator.export_results_to_dataframe(results)
#     print(f"\nDataFrame shape: {df.shape}")
#     print("\nDataFrame preview:")
#     print(df[['original_word', 'confidence_score', 'matched_alias', 'concept_id']].to_string())
    
#     # Save to CSV (optional)
#     # df.to_csv('umls_confidence_results.csv', index=False)
#     # print("\nResults saved to 'umls_confidence_results.csv'")

# if __name__ == "__main__":
#     main()








# import en_core_sci_sm
# import time
# from scispacy.linking import EntityLinker
# # Load scispaCy UMLS model
# print("Loading scispaCy model...")
# start_time = time.time()
# nlp_umls = en_core_sci_sm.load()
# nlp_umls.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls", "max_entities_per_mention": 1})
# linker_umls = nlp_umls.get_pipe("scispacy_linker")
# print("Model loaded in", time.time() - start_time, "seconds\n")

# def get_umls_confidence_score(entity):
#     candidates = linker_umls.candidate_generator([entity], k=1)
    
#     if candidates and candidates[0]:
#         top_candidate = candidates[0][0]
#         return {
#             "Entity": entity,
#             "UMLS_CUI": top_candidate.concept_id,
#             "UMLS_Name": top_candidate.aliases[0] if top_candidate.aliases else None,
#             "Confidence_Score": top_candidate.similarities[0]
#         }
#     else:
#         return {
#             "Entity": entity,
#             "UMLS_CUI": None,
#             "UMLS_Name": None,
#             "Confidence_Score": 0.0
#         }

# # Your terms
# terms = ["AKT1", "PKC-β","PKB","Protein kinase C (PKC)"]

# # Compute scores
# results = [get_umls_confidence_score(term) for term in terms]

# # Display result
# for res in results:
#     print(f"Entity: {res['Entity']}")
#     print(f" → UMLS CUI: {res['UMLS_CUI']}")
#     print(f" → UMLS Name: {res['UMLS_Name']}")
#     print(f" → Confidence Score: {res['Confidence_Score']}\n")





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

def get_umls_info(term):
    """Get UMLS details using scispaCy linker and API for semantic type."""
    try:
        candidates = linker_umls.candidate_generator([term], k=1)
        if candidates and candidates[0]:
            top_candidate = candidates[0][0]
            score = round(top_candidate.similarities[0], 4)
            alias_matched = top_candidate.aliases[0] if top_candidate.aliases else ""
            concept_id = top_candidate.concept_id
            umls_entity_data = linker_umls.kb.cui_to_entity[concept_id]

            return {
                "OriginalName": term,
                "UMLS_CUI": concept_id,
                "Matched_Alias": alias_matched,
                "Match_Score": score,
                "Canonical_Name": umls_entity_data.canonical_name,
                #"Definition": umls_entity_data.definition,
                #"Aliases": ", ".join(umls_entity_data.aliases[:5]) if umls_entity_data.aliases else "",
                "UMLS_SemanticType": get_umls_semantic_type(concept_id)
            }
    except Exception as e:
        print(f"Error processing term '{term}': {e}")

    return {
        "OriginalName": term,
        "UMLS_CUI": None,
        "Matched_Alias": "",
        "Match_Score": 0.0,
        "Canonical_Name": "",
        #"Definition": "",
       # "Aliases": "",
        "UMLS_SemanticType": None
    }

# Terms to analyze
terms_to_check = ["AKT1", "akt1", "PKC-β", "PKC-beta","PKC-α","protein kinase C-β","protein kinase C beta"]
results = [get_umls_info(term) for term in terms_to_check]

# Convert to DataFrame
df_result = pd.DataFrame(results)

# Show result
#df_result.to_string(index=False)

print(df_result)