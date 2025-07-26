# import re
# import torch
# from gliner import GLiNER
# from transformers import AutoTokenizer, AutoModelForSequenceClassification

# # ========== Load Models ==========
# print("Loading GLiNER model...")
# gliner_model = GLiNER.from_pretrained("knowledgator/gliner-x-large")
# gliner_tokenizer = AutoTokenizer.from_pretrained("knowledgator/gliner-x-large")

# print("Loading RE model...")
# re_model_name = "sagteam/pharm-relation-extraction"
# re_tokenizer = AutoTokenizer.from_pretrained(re_model_name)
# re_model = AutoModelForSequenceClassification.from_pretrained(re_model_name)
# print("RE Labels:", re_model.config.id2label)

# # Use GPU if available
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# re_model.to(device)

# # ========== Relation Label Map ==========
# if hasattr(re_model.config, 'id2label') and re_model.config.id2label:
#     relation_label_map = re_model.config.id2label
# else:
#     relation_label_map = {
#         0: "no_relation",
#         1: "ADR–Drugname",
#         2: "Drugname–Diseasename",
#         3: "Drugname–SourceInfoDrug",
#         4: "Diseasename–Indication"
#     }

# # ========== Define Input ==========
# text = """ Coartem Baby is used to treat malaria.
# """

# # ========== Extract Entities ==========
# custom_labels = ["Drug", "Disease"]
# entities = gliner_model.predict_entities(
#     text,
#     custom_labels,
#     threshold=0.01,  # relaxed threshold
#     flat_ner=True
# )

# # Filter high-confidence Drug and Disease entities
# drugs = sorted(set(ent["text"] for ent in entities if ent["label"] == "Drug" and ent["score"] >= 0.70))
# diseases = sorted(set(ent["text"] for ent in entities if ent["label"] == "Disease" and ent["score"] >= 0.70))

# print("\nExtracted Drugs:", drugs)
# print("Extracted Diseases:", diseases)

# # ========== Predict Drug–Disease Relations ==========
# def predict_relation(sentence, entity1, entity2):
#     # Proper sentence pair format
#     inputs = re_tokenizer(sentence, entity1, entity2, return_tensors="pt", truncation=True, max_length=512)

#     # Ensure inference mode only
#     inputs = {k: v.to(device) for k, v in inputs.items() if k in ["input_ids", "attention_mask", "token_type_ids"]}

#     with torch.no_grad():
#         logits = re_model(**inputs).logits
#         pred_idx = torch.argmax(logits, dim=1).item()
#         return relation_label_map.get(pred_idx, f"unknown_{pred_idx}")

# # ========== Run RE ==========
# relations = []
# for drug in drugs:
#     for disease in diseases:
#         rel1 = predict_relation(text, drug, disease)
#         rel2 = predict_relation(text, disease, drug)

#         if "Drugname–Diseasename" in [rel1, rel2]:
#             chosen = rel1 if rel1 != "no_relation" else rel2
#             relations.append(f"{drug} -> {disease} ({chosen})")

# print("\nDrug–Disease Relations Found:")
# for rel in relations:
#     print("-", rel)

# if not relations:
#     print("- No drug–disease relations found.")











# from transformers import pipeline

# # Load classifier
# classifier = pipeline("zero-shot-classification", model="knowledgator/comprehend_it-base")

# # Input sentence
# sentence = """Novartis and a Swiss nonprofit have made history, scoring the world’s first approval for a medicine to treat babies who are infected with malaria. Switzerland’s health regulator has signed off on Coartem Baby for infants who weigh between 2 and 5 kilograms (4.4 and 11 pounds) and have contracted the deadly mosquito-borne disease. Coartem Baby was endorsed by Swissmedic under a prearranged Marketing Authorization for Global Health Products (MAGHP) procedure, which will facilitate rapid approvals in eight African nations, Novartis said in a July 8 press release. The drugmaker plans to introduce the treatment on a “largely not-for-profit basis to increase access in areas where malaria is endemic," Novartis added. “For more than three decades, we have stayed the course in the fight against malaria, working relentlessly to deliver scientific breakthroughs where they are needed most,” Vas Narasimhan, Novartis’ CEO, said in the release. The pharma giant collaborated with Switzerland’s Medicines for Malaria Venture (MMV), a health-equity initiative established in 1999 to expand the use of antimalarials and innovate new compounds. Novartis has four next-generation malaria medicines in its portfolio—including one being tested in a phase 3 trial—to take on the increased resistance to existing treatments. “Malaria is one of the world’s deadliest diseases, particularly among children," Martin Fitchet, the CEO of MMV, said in the release. "But with the right resources and focus, it can be eliminated. The approval of Coartem Baby provides a necessary medicine with an optimized dose to treat an otherwise neglected group of patients.” The nod is for a new formulation of Coartem, which is also known commercially in some countries as Riamet, and was launched by Novartis in 1999. It was approved by the FDA ten years later. It is a combination of antivirals artemether and lumefantrine. Coartem Baby is dissolvable, can be taken with breast milk and has a “sweet cherry flavor," Novartis said. The approval was based on a phase 2/3 study, which investigated a new dose of Coartem to “account for metabolic differences in babies under 5 kilograms,” Novartis said in the release. RELATED GSK, Bharat Biotech to slash the price of first malaria vaccine to less than $5 per dose by 2028 Countries that participated in the assessment and which are expected to endorse the treatment are Burkina Faso, Cote d’Ivoire, Kenya, Malawi, Mozambique, Nigeria, Tanzania and Uganda. There were 263 million cases of malaria in 2023 and 597,000 deaths, nearly all of them in Africa, according to a recent World Health Organization report. Of the fatalities, 76% were children under age 5. FIERCE PHARMA WEEK The premier event bringing together over 2,200 Pharma leaders from marketing, medical affairs, PR & Communications and Commercialization. Sep. 8-11, 2025 | Philadelphia Pennsylvania Convention Center Register Now “The available malaria treatments have only been properly tested in children aged at least 6 months because smaller infants are usually excluded from treatment trials,” Umberto D'Alessandro, of The Gambia at the London School of Hygiene and Tropical Medicine, said in a release. “Neonates and young infants have immature liver function and metabolize some medicines differently, so the dose for older children may not be appropriate for small babies.” While several other large pharma companies have backed away from the field of Neglected Tropical Diseases (NTDs) in recent years, Novartis has nearly doubled its R&D spending on NTDs, advancing 10 new treatments. The company is also pursuing treatments for dengue fever, Chagas disease, leishmaniasas and cryptosporidiosis."""

# # Known entities (comma-separated)
# drug_str = "Coartem Baby, Riamet"
# disease_str = "Chagas disease, Malaria, cryptosporidiosis, dengue fever, leishmaniasas, malaria"

# # Convert to lists
# drug_list = [d.strip() for d in drug_str.split(",")]
# disease_list = [d.strip() for d in disease_str.split(",")]

# # Candidate relation labels
# candidate_labels = [
#     "treats",
#     "prevents",
#     "causes",
#     "associated_with",
#     "used_for",
#     "contraindicated_for",
#     "indicated_for",
#     "ineffective_for",
#     "no_relation"
# ]

# # Run classification for each (drug, disease) pair
# for drug in drug_list:
#     for disease in disease_list:
#         result = classifier(sentence, candidate_labels)
#         top_label = result['labels'][0]
#         top_score = result['scores'][0]

#         # Only print meaningful relations (filter out "no_relation" or low confidence)
#         if top_label != "no_relation" and top_score > 0.5:
#             print(f"{drug} {top_label} {disease} (Confidence: {top_score:.2f})")







# from transformers import pipeline

# # Load a model optimized for instructions
# prompt_model = pipeline("text2text-generation", model="google/flan-t5-base")

# def extract_relation(text, drug, disease):
#     prompt = f"""
#     Identify the relationship between '{drug}' and '{disease}' in this text:
#     Text: "{text}"
#     Possible relations: [treats, causes, prevents, inhibits, no_relation]
#     Answer:
#     """
    
#     result = prompt_model(prompt, max_length=50)
#     return result[0]['generated_text']

# # Example
# text = """Novartis and a Swiss nonprofit have made history, scoring the world’s first approval for a medicine to treat babies who are infected with malaria. Switzerland’s health regulator has signed off on Coartem Baby for infants who weigh between 2 and 5 kilograms (4.4 and 11 pounds) and have contracted the deadly mosquito-borne disease. Coartem Baby was endorsed by Swissmedic under a prearranged Marketing Authorization for Global Health Products (MAGHP) procedure, which will facilitate rapid approvals in eight African nations, Novartis said in a July 8 press release. The drugmaker plans to introduce the treatment on a “largely not-for-profit basis to increase access in areas where malaria is endemic," Novartis added. “For more than three decades, we have stayed the course in the fight against malaria, working relentlessly to deliver scientific breakthroughs where they are needed most,” Vas Narasimhan, Novartis’ CEO, said in the release. The pharma giant collaborated with Switzerland’s Medicines for Malaria Venture (MMV), a health-equity initiative established in 1999 to expand the use of antimalarials and innovate new compounds. Novartis has four next-generation malaria medicines in its portfolio—including one being tested in a phase 3 trial—to take on the increased resistance to existing treatments. “Malaria is one of the world’s deadliest diseases, particularly among children," Martin Fitchet, the CEO of MMV, said in the release. "But with the right resources and focus, it can be eliminated. The approval of Coartem Baby provides a necessary medicine with an optimized dose to treat an otherwise neglected group of patients.” The nod is for a new formulation of Coartem, which is also known commercially in some countries as Riamet, and was launched by Novartis in 1999. """
# print(extract_relation(text, "Coartem Baby", "malaria"))






# import re
# from gliner import GLiNER
# from transformers import AutoTokenizer
# import nltk
# nltk.download('punkt')
# from nltk.tokenize import sent_tokenize
# import time

# # Load GLiNER model and tokenizer
# gliner_model = GLiNER.from_pretrained("knowledgator/gliner-x-large")
# tokenizer = AutoTokenizer.from_pretrained("knowledgator/gliner-x-large")

# custom_labels = ["Drug", "Disease", "Organization", "Country"]

# # Token-aware chunking
# def chunk_text_by_tokens(text, tokenizer, max_tokens=512, overlap=20):
#     words = text.split()
#     chunks = []
#     start = 0
#     while start < len(words):
#         end = len(words)
#         for i in range(start + max_tokens, start, -1):
#             chunk = " ".join(words[start:i])
#             tokenized = tokenizer(chunk, truncation=False, return_tensors="pt")
#             if tokenized.input_ids.shape[1] <= max_tokens:
#                 end = i
#                 break
#         chunks.append(" ".join(words[start:end]))
#         start = end - overlap if end - overlap > start else end
#     return chunks

# # 🔹 Input your title and description (single example)
# title = "Novartis gains approval for Coartem Baby"
# description = "Coartem Baby is used to treat malaria in infants under 5 kg. Novartis announced the approval and plans to distribute the drug across multiple African countries."

# full_text = f"{title}\n{description}"

# # Start timer
# start_time = time.time()

# chunks = chunk_text_by_tokens(full_text, tokenizer, max_tokens=512, overlap=20)
# all_entities = []
# sentence_entity_map = {}

# # NER + sentence-level mapping
# for chunk in chunks:
#     try:
#         sentences = sent_tokenize(chunk)
#         for sentence in sentences:
#             entities = gliner_model.predict_entities(
#                 sentence,
#                 custom_labels,
#                 threshold=0.20,
#                 flat_ner=True
#             )
#             all_entities.extend(entities)
#             sentence_entity_map[sentence] = [e for e in entities]
#     except Exception as e:
#         print(f"❌ Error: {e}")

# # Extract entities
# drug_list = sorted(set(ent["text"] for ent in all_entities if ent["label"] == "Drug"))
# disease_list = sorted(set(ent["text"] for ent in all_entities if ent["label"] == "Disease"))
# org_list = sorted(set(ent["text"] for ent in all_entities if ent["label"] == "Organization"))
# country_list = sorted(set(ent["text"] for ent in all_entities if ent["label"] == "Country"))

# # Drug–Disease relation extraction
# drug_disease_relations = set()
# for sentence, entities in sentence_entity_map.items():
#     drugs_in_sent = [e["text"] for e in entities if e["label"] == "Drug"]
#     diseases_in_sent = [e["text"] for e in entities if e["label"] == "Disease"]
#     for drug in drugs_in_sent:
#         for disease in diseases_in_sent:
#             drug_disease_relations.add(f"{drug} -> {disease}")

# # Print results
# print("\n🧪 Extracted Entities and Relations:")
# print(f"Drugs: {', '.join(drug_list) if drug_list else 'None'}")
# print(f"Diseases: {', '.join(disease_list) if disease_list else 'None'}")
# print(f"Organizations: {', '.join(org_list) if org_list else 'None'}")
# print(f"Countries: {', '.join(country_list) if country_list else 'None'}")
# print(f"Drug-Disease Relations: {'; '.join(sorted(drug_disease_relations)) if drug_disease_relations else 'None'}")

# print(f"\n✅ Completed in {time.time() - start_time:.2f} seconds.")







# import pandas as pd
# import spacy

# # Load SpaCy for sentence tokenization
# nlp = spacy.load("en_core_web_sm")

# # Load your extracted entities CSV
# input_path = "C:/Users/EZB-VM-06/Downloads/070_gliner_largeModel_1028 (1).xlsx"
# df = pd.read_excel(input_path)

# results = []

# def normalize_list(val):
#     return [v.strip().lower() for v in str(val).split(",") if v.strip()]

# for idx, row in df.iterrows():
#     description = row.get("description", "")
#     title = row.get("title", "")
    
#     # Entities
#     drugs = normalize_list(row.get("drugs", ""))
#     diseases = normalize_list(row.get("diseases", ""))
#     companies = normalize_list(row.get("organizations", ""))

#     doc = nlp(description)
#     for sent in doc.sents:
#         sent_text = sent.text.lower()

#         found_drugs = [d for d in drugs if d in sent_text]
#         found_diseases = [di for di in diseases if di in sent_text]
#         found_companies = [c for c in companies if c in sent_text]

#         # Create all meaningful combinations
#         for drug in found_drugs:
#             for disease in found_diseases:
#                 for company in found_companies:
#                     results.append({
#                         "drug": drug.title(),
#                         "disease": disease.title(),
#                         "company": company.title(),
#                         "sentence": sent.text.strip(),
#                         "correlation_type": "drug–disease–company",
#                         "title": title,
#                         "row_index": idx
#                     })

#             if not found_diseases:
#                 for company in found_companies:
#                     results.append({
#                         "drug": drug.title(),
#                         "disease": "",
#                         "company": company.title(),
#                         "sentence": sent.text.strip(),
#                         "correlation_type": "drug–company",
#                         "title": title,
#                         "row_index": idx
#                     })

#             for disease in found_diseases:
#                 results.append({
#                     "drug": drug.title(),
#                     "disease": disease.title(),
#                     "company": "",
#                     "sentence": sent.text.strip(),
#                     "correlation_type": "drug–disease",
#                     "title": title,
#                     "row_index": idx
#                 })

#         for disease in found_diseases:
#             for company in found_companies:
#                 results.append({
#                     "drug": "",
#                     "disease": disease.title(),
#                     "company": company.title(),
#                     "sentence": sent.text.strip(),
#                     "correlation_type": "disease–company",
#                     "title": title,
#                     "row_index": idx
#                 })

# # Save the output
# output_df = pd.DataFrame(results)
# output_df.to_csv("C:/Users/EZB-VM-06/Downloads/contextual_correlations.csv", index=False)

# print("✅ Saved correlations to: contextual_correlations.csv")





# from transformers import AutoTokenizer, BioGptForCausalLM
# import torch
# import re

# # Load the model and tokenizer
# print("Loading BioGPT model...")
# tokenizer = AutoTokenizer.from_pretrained("microsoft/biogpt")
# model = BioGptForCausalLM.from_pretrained("microsoft/biogpt")

# # Set pad_token_id to eos_token_id if not already set
# if tokenizer.pad_token is None:
#     tokenizer.pad_token = tokenizer.eos_token

# # Your input text
# text = """
# Ponsegromab (PF-06946860) is a promising therapeutic candidate developed by Pfizer for the treatment of cancer cachexia. 
# Cancer cachexia is associated with systemic inflammation and contributes to poor treatment response. 
# Dexamethasone may alleviate appetite loss in cancer cachexia.
# """

# # Pre-extracted entities from GLiNER (example based on your document)
# drugs = ["Ponsegromab", "PF-06946860", "Dexamethasone"]
# diseases = ["cancer cachexia", "systemic inflammation", "appetite loss", "poor treatment response"]
# organizations = ["Pfizer"]

# # Create a more focused prompt for relation extraction
# def create_relation_prompt(text, drugs, diseases, organizations):
#     entities_str = f"Drugs: {', '.join(drugs)}\nDiseases/Conditions: {', '.join(diseases)}\nOrganizations: {', '.join(organizations)}"
    
#     prompt = f"""Given the following biomedical text and identified entities, extract relationships between them.

# Text: {text.strip()}

# Identified Entities:
# {entities_str}

# Task: Find relationships between these entities using these relation types:
# - treats (drug treats disease)
# - causes (entity causes condition)  
# - associated_with (entity associated with condition)
# - developed_by (drug developed by organization)
# - alleviates (drug alleviates symptom)

# Format each relation as: Entity1 -> relation -> Entity2

# Relations:
# """
#     return prompt

# # Generate relations using BioGPT
# def extract_relations_biogpt(text, drugs, diseases, organizations):
#     prompt = create_relation_prompt(text, drugs, diseases, organizations)
    
#     # Tokenize input
#     inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    
#     # Generate with optimized parameters
#     with torch.no_grad():
#         outputs = model.generate(
#             **inputs,
#             max_new_tokens=150,
#             min_length=len(inputs['input_ids'][0]) + 20,
#             do_sample=True,
#             temperature=0.3,  # Lower temperature for more focused output
#             top_p=0.85,
#             top_k=40,
#             repetition_penalty=1.3,  # Penalize repetition
#             no_repeat_ngram_size=3,  # Prevent 3-gram repetition
#             pad_token_id=tokenizer.eos_token_id,
#             eos_token_id=tokenizer.eos_token_id,
#             early_stopping=True
#         )
    
#     # Decode and extract only the generated part
#     decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     extracted = decoded[len(prompt):].strip()
    
#     return extracted

# # Alternative approach with simpler prompting
# def extract_relations_simple(text, drugs, diseases):
#     # Create simpler, more direct prompts for each drug
#     relations = []
    
#     for drug in drugs:
#         simple_prompt = f"""The drug {drug} is mentioned in this text: "{text.strip()}"

# What does {drug} treat or affect?
# Answer: {drug}"""
        
#         inputs = tokenizer(simple_prompt, return_tensors="pt", truncation=True, max_length=256)
        
#         with torch.no_grad():
#             outputs = model.generate(
#                 **inputs,
#                 max_new_tokens=50,
#                 do_sample=True,
#                 temperature=0.2,
#                 repetition_penalty=1.4,
#                 no_repeat_ngram_size=2,
#                 pad_token_id=tokenizer.eos_token_id,
#                 early_stopping=True
#             )
        
#         decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
#         extracted = decoded[len(simple_prompt):].strip()
        
#         if extracted and len(extracted) > 5:  # Filter out very short responses
#             relations.append(f"{drug} -> treats/affects -> {extracted}")
    
#     return relations

# # Rule-based relation extraction as backup
# def extract_relations_rules(text, drugs, diseases, organizations):
#     relations = []
#     text_lower = text.lower()
    
#     # Define relation patterns
#     relation_patterns = {
#         'treats': ['treatment of', 'treat', 'therapy for', 'therapeutic for'],
#         'developed_by': ['developed by', 'by'],
#         'associated_with': ['associated with', 'linked to', 'related to'],
#         'causes': ['causes', 'leads to', 'results in'],
#         'alleviates': ['alleviate', 'reduce', 'relieve']
#     }
    
#     # Extract drug-disease relations
#     for drug in drugs:
#         for disease in diseases:
#             # Check for treatment relations
#             if any(pattern in text_lower for pattern in relation_patterns['treats']):
#                 if drug.lower() in text_lower and disease.lower() in text_lower:
#                     relations.append(f"{drug} -> treats -> {disease}")
            
#             # Check for alleviation relations
#             if any(pattern in text_lower for pattern in relation_patterns['alleviates']):
#                 if drug.lower() in text_lower and disease.lower() in text_lower:
#                     relations.append(f"{drug} -> alleviates -> {disease}")
    
#     # Extract drug-organization relations
#     for drug in drugs:
#         for org in organizations:
#             if any(pattern in text_lower for pattern in relation_patterns['developed_by']):
#                 if drug.lower() in text_lower and org.lower() in text_lower:
#                     relations.append(f"{drug} -> developed_by -> {org}")
    
#     # Extract disease associations
#     for i, disease1 in enumerate(diseases):
#         for disease2 in diseases[i+1:]:
#             if any(pattern in text_lower for pattern in relation_patterns['associated_with']):
#                 if disease1.lower() in text_lower and disease2.lower() in text_lower:
#                     relations.append(f"{disease1} -> associated_with -> {disease2}")
    
#     return list(set(relations))  # Remove duplicates

# # Main execution
# print("🔎 Extracting Relations using BioGPT...")
# print("="*60)

# # Method 1: BioGPT with structured prompt
# print("Method 1: BioGPT Structured Extraction")
# biogpt_relations = extract_relations_biogpt(text, drugs, diseases, organizations)
# print(biogpt_relations)
# print()

# # Method 2: BioGPT with simple prompts
# print("Method 2: BioGPT Simple Prompts")
# simple_relations = extract_relations_simple(text, drugs, diseases)
# for relation in simple_relations:
#     print(f"• {relation}")
# print()



# # Combine and deduplicate results
# print("="*60)
# print("🎯 FINAL EXTRACTED RELATIONS:")
# print("="*60)

# # Automatically extract relations from text using pattern matching
# def auto_extract_relations(text, drugs, diseases, organizations):
#     """Automatically extract relations from any text"""
#     relations = set()  # Use set to avoid duplicates
#     text_lower = text.lower()
    
#     # Split text into sentences for better analysis
#     sentences = text.replace('.', '.|').replace('!', '!|').replace('?', '?|').split('|')
#     sentences = [s.strip() for s in sentences if s.strip()]
    
#     for sentence in sentences:
#         sentence_lower = sentence.lower()
        
#         # Find entities present in this sentence
#         sentence_drugs = [drug for drug in drugs if drug.lower() in sentence_lower]
#         sentence_diseases = [disease for disease in diseases if disease.lower() in sentence_lower]
#         sentence_orgs = [org for org in organizations if org.lower() in sentence_lower]
        
#         # Extract drug-disease relations
#         for drug in sentence_drugs:
#             for disease in sentence_diseases:
#                 # Treatment relations
#                 if any(word in sentence_lower for word in ['treatment', 'treat', 'therapy', 'therapeutic']):
#                     relations.add(f"{drug} -> treats -> {disease}")
                
#                 # Alleviation relations  
#                 elif any(word in sentence_lower for word in ['alleviate', 'relieve', 'reduce', 'help']):
#                     relations.add(f"{drug} -> alleviates -> {disease}")
                
#                 # General association if entities co-occur
#                 else:
#                     relations.add(f"{drug} -> associated_with -> {disease}")
        
#         # Extract drug-organization relations
#         for drug in sentence_drugs:
#             for org in sentence_orgs:
#                 if any(word in sentence_lower for word in ['developed', 'made', 'created', 'produced']):
#                     relations.add(f"{drug} -> developed_by -> {org}")
#                 elif any(word in sentence_lower for word in ['by', 'from']):
#                     relations.add(f"{drug} -> developed_by -> {org}")
        
#         # Extract disease associations
#         if len(sentence_diseases) >= 2:
#             for i, disease1 in enumerate(sentence_diseases):
#                 for disease2 in sentence_diseases[i+1:]:
#                     if any(word in sentence_lower for word in ['associated', 'linked', 'related', 'connected']):
#                         relations.add(f"{disease1} -> associated_with -> {disease2}")
#                     elif any(word in sentence_lower for word in ['causes', 'leads to', 'results in', 'contributes to']):
#                         relations.add(f"{disease1} -> causes -> {disease2}")
    
#     return list(relations)

# # Automatically extract relations
# auto_relations = auto_extract_relations(text, drugs, diseases, organizations)

# # Combine results from all methods
# all_relations = set()

# # Add automatically extracted relations
# all_relations.update(auto_relations)

# # Try to parse BioGPT output for valid relations
# if biogpt_relations:
#     biogpt_lines = biogpt_relations.split('\n')
#     for line in biogpt_lines:
#         line = line.strip()
#         if '->' in line and len(line.split('->')) == 3:
#             all_relations.add(line)

# # Display final relations
# final_relations = sorted(list(all_relations))
# for i, relation in enumerate(final_relations, 1):
#     print(f"{i}. {relation}")

# # Function to format relations for downstream use
# def format_relations_for_export(relations):
#     """Format relations for easy integration with other systems"""
#     formatted = []
#     for relation in relations:
#         parts = relation.split(' -> ')
#         if len(parts) == 3:
#             entity1, relation_type, entity2 = parts
#             formatted.append({
#                 'entity1': entity1.strip(),
#                 'relation': relation_type.strip(), 
#                 'entity2': entity2.strip(),
#                 'confidence': 'auto_extracted'  # Indicates automatic extraction
#             })
#     return formatted

# # Export formatted relations
# print("\n📊 Formatted Relations for Export:")
# formatted_relations = format_relations_for_export(final_relations)
# for rel in formatted_relations:
#     print(f"  {rel}")















from transformers import AutoTokenizer, BioGptForCausalLM
import torch
import re
from typing import List, Dict, Set

# Load the model and tokenizer
print("Loading BioGPT model...")
tokenizer = AutoTokenizer.from_pretrained("microsoft/biogpt")
model = BioGptForCausalLM.from_pretrained("microsoft/biogpt")

# Set pad_token_id to eos_token_id if not already set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Your long cancer cachexia market text
long_text = """
UK authority cracks down on weight-loss drug ads
A series of adverts for weight-loss services have been banned by the Advertising Standards Authority in the UK for promoting prescription-only medicines (POMs), which is against the law.
Nine ads – including a social post by TV personality and social media influencer Gemma Collins – have been taken to task as part of a crackdown on the illegal promotion of GLP-1 agonist injections for weight-loss to the public.
All GLP-1 medicines are POM, which means they can only be prescribed by a healthcare professional, and only some are approved for weight loss in the UK.
Burgeoning demand for the injections – sometimes known as 'skinny jabs' – has raised concerns of people being exposed to counterfeit and otherwise illegal preparations that could according to the UK medicines regulatory authority, the MHRA, "pose significant health risks."
In Collins' case, the ASA took exception to a paid-for Instagram post in January in which she promoted the Swedish digital health platform' Yazen's weight-loss service, saying: "I'm starting this year two sizes down, thanks to Yazen's weight loss app and medication. It's really quick and easy to get started with Yazen, it has absolutely changed my life."
She continued: "I finally found something that actually I lose weight on. All you need to do is download the app and answer a few quick questions about your goals. Do not buy it from anyone but Yazen."
Yazen said it does not sell medicine directly and has no interest in advertising POMs to the public. However, the ASA said the MHRA expressed concern that consumers were likely to be led to request a POM for weight loss, and it has concluded a breach of its code of practice had been committed.
Collins said she accepted her posts had promoted Yazen and its app, and she would follow the ASA's guidance in future.
While Collins' involvement has given the Yazen ad prominence, eight other adverts from other providers have also been banned for promoting prescription-only weight-loss medications.
Those include paid-for ads on Instagram and Facebook parent Meta for CheqUp Health, HealthExpress.co.uk, and Cloud Pharmacy, and Google search ads sponsored by Juniper UK, Phlo Clinic, SemaPen, and PharmacyOnline.co.uk.
"Using our AI-based Active Ad Monitoring system to monitor for problem ads and launch proactive investigations, we have published rulings that make crystal clear that all injectable forms of weight-loss medication are POMs and can't be advertised, even where ads don't explicitly name a medicine," said the ASA.
"We're not here to regulate the drugs, their safety or availability. And we acknowledge the role they might play in combating obesity. But we are here to protect people from irresponsible and, in the case of weight-loss POMs, illegal ads," it added.
"These issues won't be resolved overnight, we know the scale of the problem and that means approaching this in a methodical, phased way. We'll continue to make sure weight-loss providers take their medicine rather than advertising it."
It's worth noting that some social media companies have also started to pay attention to the promotion of POMs, perhaps concerned they could come under scrutiny by the authorities.
For example, TikTok recently changed its terms on branded content to prohibit paid-for posts involving a range of industries, including "pharmaceuticals, healthcare, and medicine products" claiming health benefits.
Photo by Brett Jordan on Unsplash

"""

# Comprehensive entity lists extracted from the document
drugs = [
    "POMs"
]

diseases = [
    "obesity's"
    # "Alzheimer's",
    # "cancer",
    # "diabetes",
    # "rare diseases",

]

organizations = [
   "dvertising Standards Authority",
    "CheqUp Health",
    "Cloud Pharmacy",
    "Phlo Clinic"
]

def chunk_text(text: str, max_chunk_size: int = 800, overlap: int = 100) -> List[str]:
    """
    Split long text into overlapping chunks for processing
    """
    sentences = text.replace('.', '.|').replace('!', '!|').replace('?', '?|').split('|')
    sentences = [s.strip() for s in sentences if s.strip()]
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk + sentence) > max_chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            # Keep last few sentences for overlap
            overlap_sentences = current_chunk.split('.')[-2:]
            current_chunk = '. '.join(overlap_sentences) + '. ' + sentence
        else:
            current_chunk += sentence + '. '
    
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks

def extract_entities_from_chunk(chunk: str, drugs: List[str], diseases: List[str], organizations: List[str]) -> Dict[str, List[str]]:
    """
    Extract entities present in a specific chunk
    """
    chunk_lower = chunk.lower()
    
    found_entities = {
        'drugs': [drug for drug in drugs if drug.lower() in chunk_lower],
        'diseases': [disease for disease in diseases if disease.lower() in chunk_lower],
        'organizations': [org for org in organizations if org.lower() in chunk_lower]
    }
    
    return found_entities

def create_focused_prompt(chunk: str, found_entities: Dict[str, List[str]]) -> str:
    """
    Create a focused prompt for a specific chunk with its entities - enhanced for precision
    """
    if not any(found_entities.values()):
        return None
    
    entities_str = ""
    if found_entities['drugs']:
        entities_str += f"Drugs: {', '.join(found_entities['drugs'][:8])}\n"
    if found_entities['diseases']:
        entities_str += f"Diseases: {', '.join(found_entities['diseases'][:8])}\n"
    if found_entities['organizations']:
        entities_str += f"Organizations: {', '.join(found_entities['organizations'][:6])}\n"
    
    prompt = f"""Extract precise biomedical relationships from this text. Focus on explicit relationships mentioned in the text.

Text: {chunk[:600]}

Entities in this text:
{entities_str}

Extract ONLY relationships that are explicitly stated in the text using these specific types:
- approved_for: drug is explicitly approved for treating a disease (e.g., "approved for treatment of")
- treats: drug explicitly treats or is used for treatment of a disease
- developed_by: drug is developed/made by an organization
- causes: one condition directly causes another
- alleviates: drug reduces or helps with symptoms
- associated_with: general association (use only if no specific relationship applies)

IMPORTANT: 
- Use "approved_for" when text mentions approval, licensing, or regulatory authorization
- Use "treats" when text mentions treatment, therapy, or therapeutic use
- Be precise - don't guess relationships not clearly stated in the text

Format each relationship as: Entity1 -> relation_type -> Entity2

Relationships found:
"""
    return prompt

def extract_relations_from_chunk(chunk: str, found_entities: Dict[str, List[str]]) -> List[str]:
    """
    Extract relations from a single chunk using BioGPT
    """
    prompt = create_focused_prompt(chunk, found_entities)
    if not prompt:
        return []
    
    try:
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=400)
        
        # Generate with optimized parameters for relation extraction
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=120,
                min_length=len(inputs['input_ids'][0]) + 10,
                do_sample=True,
                temperature=0.3,
                top_p=0.85,
                top_k=40,
                repetition_penalty=1.3,
                no_repeat_ngram_size=3,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                early_stopping=True
            )
        
        # Decode and extract relations
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        extracted = decoded[len(prompt):].strip()
        
        # Parse relations from the output
        relations = []
        for line in extracted.split('\n'):
            line = line.strip()
            if '->' in line and len(line.split('->')) >= 2:
                # Clean and validate the relation
                parts = [part.strip() for part in line.split('->')]
                if len(parts) >= 3:
                    relations.append(' -> '.join(parts[:3]))
        
        return relations
    
    except Exception as e:
        print(f"Error processing chunk: {e}")
        return []

def extract_relations_rule_based(chunk: str, found_entities: Dict[str, List[str]]) -> Set[str]:
    """
    Enhanced rule-based relation extraction with more precise pattern matching
    """
    relations = set()
    chunk_lower = chunk.lower()
    
    # Define more specific relation patterns with context
    treatment_patterns = {
        'treats': ['treatment of', 'treat', 'therapy for', 'therapeutic for', 'used to treat', 'indicated for'],
        'approved_for': ['approved for', 'approved in', 'licensed for', 'approved for the treatment of'],
        'alleviates': ['alleviate', 'relieve', 'reduce', 'help with', 'ameliorate'],
        'developed_by': ['developed by', "by", "'s", "from"],
        'causes': ['causes', 'leads to', 'results in', 'triggers', 'contributes to'],
        'associated_with': ['associated with', 'linked to', 'related to', 'connected to']
    }
    
    # Extract drug-disease relations with priority order (most specific first)
    for drug in found_entities['drugs']:
        for disease in found_entities['diseases']:
            if drug.lower() in chunk_lower and disease.lower() in chunk_lower:
                relation_found = False
                
                # Check for approval patterns first (most specific)
                for pattern in treatment_patterns['approved_for']:
                    if pattern in chunk_lower:
                        # Look for context around the approval
                        drug_pos = chunk_lower.find(drug.lower())
                        disease_pos = chunk_lower.find(disease.lower())
                        pattern_pos = chunk_lower.find(pattern)
                        
                        # Check if drug, approval pattern, and disease appear in logical order
                        if abs(drug_pos - pattern_pos) < 100 and abs(pattern_pos - disease_pos) < 100:
                            relations.add(f"{drug} -> approved_for -> {disease}")
                            relation_found = True
                            break
                
                # Check for treatment patterns
                if not relation_found:
                    for pattern in treatment_patterns['treats']:
                        if pattern in chunk_lower:
                            drug_pos = chunk_lower.find(drug.lower())
                            disease_pos = chunk_lower.find(disease.lower())
                            pattern_pos = chunk_lower.find(pattern)
                            
                            if abs(drug_pos - pattern_pos) < 80 and abs(pattern_pos - disease_pos) < 80:
                                relations.add(f"{drug} -> treats -> {disease}")
                                relation_found = True
                                break
                
                # Check for alleviation patterns
                if not relation_found:
                    for pattern in treatment_patterns['alleviates']:
                        if pattern in chunk_lower:
                            relations.add(f"{drug} -> alleviates -> {disease}")
                            relation_found = True
                            break
                
                # Only use generic association if no specific relation found
                if not relation_found:
                    for pattern in treatment_patterns['associated_with']:
                        if pattern in chunk_lower:
                            relations.add(f"{drug} -> associated_with -> {disease}")
                            break
    
    # Extract drug-organization relations with better context checking
    for drug in found_entities['drugs']:
        for org in found_entities['organizations']:
            if drug.lower() in chunk_lower and org.lower() in chunk_lower:
                # Check for possessive or development patterns
                if f"{org.lower()}'s {drug.lower()}" in chunk_lower or f"{org.lower()} {drug.lower()}" in chunk_lower:
                    relations.add(f"{drug} -> developed_by -> {org}")
                elif any(pattern in chunk_lower for pattern in treatment_patterns['developed_by']):
                    drug_pos = chunk_lower.find(drug.lower())
                    org_pos = chunk_lower.find(org.lower())
                    if abs(drug_pos - org_pos) < 50:  # Close proximity
                        relations.add(f"{drug} -> developed_by -> {org}")
    
    # Extract disease causation relations with better specificity
    for i, disease1 in enumerate(found_entities['diseases']):
        for disease2 in found_entities['diseases'][i+1:]:
            if disease1.lower() in chunk_lower and disease2.lower() in chunk_lower:
                # Check for direct causation
                for pattern in treatment_patterns['causes']:
                    if pattern in chunk_lower:
                        disease1_pos = chunk_lower.find(disease1.lower())
                        disease2_pos = chunk_lower.find(disease2.lower())
                        pattern_pos = chunk_lower.find(pattern)
                        
                        # Ensure logical order: disease1 -> pattern -> disease2
                        if disease1_pos < pattern_pos < disease2_pos and (pattern_pos - disease1_pos) < 50:
                            relations.add(f"{disease1} -> causes -> {disease2}")
                            break
                else:
                    # Check for association only if no causation found
                    for pattern in treatment_patterns['associated_with']:
                        if pattern in chunk_lower:
                            relations.add(f"{disease1} -> associated_with -> {disease2}")
                            break
    
    return relations

def post_process_relations(relations: Set[str], original_text: str) -> Set[str]:
    """
    Post-process relations to correct common misclassifications
    """
    corrected_relations = set()
    text_lower = original_text.lower()
    
    for relation in relations:
        parts = relation.split(' -> ')
        if len(parts) != 3:
            continue
            
        entity1, rel_type, entity2 = [part.strip() for part in parts]
        
        # Check if this should be "approved_for" instead of "associated_with" or "treats"
        if rel_type in ['associated_with', 'treats']:
            # Look for approval context in the original text
            approval_indicators = [
                'approved for', 'approved in', 'was approved', 'approval for',
                'licensed for', 'authorization for', 'indicated for'
            ]
            
            entity1_lower = entity1.lower()
            entity2_lower = entity2.lower()
            
            # Find sentences containing both entities
            sentences = text_lower.split('.')
            for sentence in sentences:
                if entity1_lower in sentence and entity2_lower in sentence:
                    # Check if approval language is present
                    if any(indicator in sentence for indicator in approval_indicators):
                        corrected_relations.add(f"{entity1} -> approved_for -> {entity2}")
                        break
            else:
                # No approval context found, keep original relation
                corrected_relations.add(relation)
        else:
            # Keep other relation types as is
            corrected_relations.add(relation)
    
    return corrected_relations

def process_long_text(text: str, drugs: List[str], diseases: List[str], organizations: List[str]) -> List[Dict]:
    """
    Main function to process long text and extract relations - enhanced version
    """
    print("🔄 Processing long text...")
    print(f"Text length: {len(text)} characters")
    
    # Step 1: Chunk the text
    chunks = chunk_text(text, max_chunk_size=800, overlap=100)
    print(f"Split into {len(chunks)} chunks")
    
    all_relations = set()
    chunk_results = []
    
    # Step 2: Process each chunk
    for i, chunk in enumerate(chunks):
        print(f"\n📝 Processing chunk {i+1}/{len(chunks)}...")
        
        # Extract entities from this chunk
        found_entities = extract_entities_from_chunk(chunk, drugs, diseases, organizations)
        
        if not any(found_entities.values()):
            print("  No entities found in this chunk, skipping...")
            continue
        
        print(f"  Found: {len(found_entities['drugs'])} drugs, {len(found_entities['diseases'])} diseases, {len(found_entities['organizations'])} organizations")
        
        # Extract relations using BioGPT
        biogpt_relations = extract_relations_from_chunk(chunk, found_entities)
        
        # Extract relations using rule-based approach
        rule_relations = extract_relations_rule_based(chunk, found_entities)
        
        # Combine relations from both methods
        chunk_relations = set(biogpt_relations) | rule_relations
        
        # Post-process to correct misclassifications
        chunk_relations = post_process_relations(chunk_relations, chunk)
        
        all_relations.update(chunk_relations)
        
        chunk_results.append({
            'chunk_id': i+1,
            'chunk_text': chunk[:200] + "..." if len(chunk) > 200 else chunk,
            'entities': found_entities,
            'relations': list(chunk_relations)
        })
        
        print(f"  Extracted {len(chunk_relations)} relations from this chunk")
    
    # Final post-processing on all relations using the full text
    print("\n🔧 Post-processing relations for accuracy...")
    all_relations = post_process_relations(all_relations, text)
    
    return list(all_relations), chunk_results

def format_relations_for_export(relations: List[str]) -> List[Dict]:
    """
    Format relations for easy integration with other systems
    """
    formatted = []
    for relation in relations:
        parts = relation.split(' -> ')
        if len(parts) >= 3:
            formatted.append({
                'entity1': parts[0].strip(),
                'relation': parts[1].strip(),
                'entity2': parts[2].strip(),
                'confidence': 'extracted',
                'source': 'biogpt_and_rules'
            })
    return formatted

# Main execution
print("🚀 Starting BioGPT Relation Extraction for Long Text")
print("=" * 80)

# Process the long text
final_relations, chunk_results = process_long_text(long_text, drugs, diseases, organizations)

print("\n" + "=" * 80)
print("🎯 FINAL EXTRACTED RELATIONS")
print("=" * 80)

# Display relations grouped by type
relation_types = {}
for relation in final_relations:
    parts = relation.split(' -> ')
    if len(parts) >= 3:
        rel_type = parts[1].strip()
        if rel_type not in relation_types:
            relation_types[rel_type] = []
        relation_types[rel_type].append(relation)

for rel_type, relations in relation_types.items():
    print(f"\n📋 {rel_type.upper()} Relations:")
    for i, relation in enumerate(sorted(relations), 1):
        print(f"  {i}. {relation}")

print(f"\n📊 SUMMARY:")
print(f"Total relations extracted: {len(final_relations)}")
print(f"Relation types found: {len(relation_types)}")
print(f"Text chunks processed: {len(chunk_results)}")

# Export formatted relations
print("\n💾 Formatted Relations for Export:")
formatted_relations = format_relations_for_export(final_relations)
for i, rel in enumerate(formatted_relations[:10], 1):  # Show first 10
    print(f"  {i}. {rel}")

if len(formatted_relations) > 10:
    print(f"  ... and {len(formatted_relations) - 10} more relations")

# Save to Excel with duplicate removal
def save_relations_to_excel(relations_list: List[Dict], filename: str = "C:/Users/EZB-VM-06/Downloads/cancer_cachexia_relations_2.xlsx"):
    """
    Save formatted relations to Excel file with duplicate removal
    """
    import pandas as pd
    from datetime import datetime
    
    try:
        # Convert to DataFrame
        df = pd.DataFrame(relations_list)
        
        # Remove duplicates based on entity1, relation, and entity2 (case-insensitive)
        print("\n🔄 Removing duplicates...")
        initial_count = len(df)
        
        # Create normalized columns for duplicate detection
        df['entity1_norm'] = df['entity1'].str.lower().str.strip()
        df['entity2_norm'] = df['entity2'].str.lower().str.strip()
        df['relation_norm'] = df['relation'].str.lower().str.strip()
        
        # Remove duplicates based on normalized columns
        df_unique = df.drop_duplicates(subset=['entity1_norm', 'relation_norm', 'entity2_norm'], 
                                     keep='first')
        
        # Drop the normalization columns
        df_unique = df_unique.drop(['entity1_norm', 'entity2_norm', 'relation_norm'], axis=1)
        
        # Add metadata
        df_unique['extraction_date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        df_unique['id'] = range(1, len(df_unique) + 1)
        
        # Reorder columns
        column_order = ['id', 'entity1', 'relation', 'entity2', 'confidence', 'source', 'extraction_date']
        df_unique = df_unique[column_order]
        
        # Sort by relation type and then by entity1
        df_unique = df_unique.sort_values(['relation', 'entity1', 'entity2'])
        df_unique['id'] = range(1, len(df_unique) + 1)  # Reassign IDs after sorting
        
        # Create summary statistics
        relation_counts = df_unique['relation'].value_counts().to_dict()
        
        # Save to Excel with multiple sheets
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # Main relations sheet
            df_unique.to_excel(writer, sheet_name='Relations', index=False)
            
            # Summary sheet
            summary_data = {
                'Metric': ['Total Relations', 'Original Relations', 'Duplicates Removed', 'Unique Entities (Entity1)', 
                          'Unique Entities (Entity2)', 'Relation Types'],
                'Count': [len(df_unique), initial_count, initial_count - len(df_unique),
                         df_unique['entity1'].nunique(), df_unique['entity2'].nunique(), 
                         df_unique['relation'].nunique()]
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Relation types breakdown
            relation_breakdown = pd.DataFrame(list(relation_counts.items()), 
                                            columns=['Relation Type', 'Count'])
            relation_breakdown.to_excel(writer, sheet_name='Relation_Types', index=False)
            
            # Entity frequency analysis
            entity1_freq = df_unique['entity1'].value_counts().head(20).reset_index()
            entity1_freq.columns = ['Entity', 'Frequency']
            entity1_freq.to_excel(writer, sheet_name='Top_Entities', index=False)
        
        print(f"✅ Relations saved to '{filename}'")
        print(f"📊 Statistics:")
        print(f"  - Original relations: {initial_count}")
        print(f"  - Unique relations: {len(df_unique)}")
        print(f"  - Duplicates removed: {initial_count - len(df_unique)}")
        print(f"  - Relation types: {len(relation_counts)}")
        
        # Show relation type breakdown
        print(f"\n📋 Relation Types:")
        for rel_type, count in sorted(relation_counts.items()):
            print(f"  - {rel_type}: {count}")
        
        return df_unique, filename
        
    except ImportError:
        print("❌ pandas and openpyxl are required for Excel export")
        print("Install with: pip install pandas openpyxl")
        return None, None
    except Exception as e:
        print(f"❌ Error saving to Excel: {e}")
        return None, None

def save_relations_to_csv(relations_list: List[Dict], filename: str = "C:/Users/EZB-VM-06/Downloads/cancer_cachexia_relations_2.csv"):
    """
    Alternative CSV export if Excel is not available
    """
    import csv
    from datetime import datetime
    
    try:
        # Remove duplicates manually
        seen = set()
        unique_relations = []
        
        for rel in relations_list:
            # Create a normalized key for duplicate detection
            key = (rel['entity1'].lower().strip(), 
                   rel['relation'].lower().strip(), 
                   rel['entity2'].lower().strip())
            
            if key not in seen:
                seen.add(key)
                rel['extraction_date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                rel['id'] = len(unique_relations) + 1
                unique_relations.append(rel)
        
        # Sort relations
        unique_relations.sort(key=lambda x: (x['relation'], x['entity1'], x['entity2']))
        
        # Reassign IDs after sorting
        for i, rel in enumerate(unique_relations, 1):
            rel['id'] = i
        
        # Write to CSV
        fieldnames = ['id', 'entity1', 'relation', 'entity2', 'confidence', 'source', 'extraction_date']
        
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(unique_relations)
        
        print(f"✅ Relations saved to '{filename}'")
        print(f"📊 Statistics:")
        print(f"  - Original relations: {len(relations_list)}")
        print(f"  - Unique relations: {len(unique_relations)}")
        print(f"  - Duplicates removed: {len(relations_list) - len(unique_relations)}")
        
        return unique_relations, filename
        
    except Exception as e:
        print(f"❌ Error saving to CSV: {e}")
        return None, None

# Save the relations
print("\n💾 Saving relations to file...")

# Try Excel first, fall back to CSV
df_saved, excel_filename = save_relations_to_excel(formatted_relations)

if df_saved is None:
    print("Falling back to CSV format...")
    unique_relations, csv_filename = save_relations_to_csv(formatted_relations)

# Show chunk-by-chunk results (optional)
print(f"\n🔍 Chunk-by-Chunk Results:")
for result in chunk_results[:3]:  # Show first 3 chunks
    print(f"\nChunk {result['chunk_id']}:")
    print(f"  Text: {result['chunk_text']}")
    print(f"  Entities: {sum(len(v) for v in result['entities'].values())} total")
    print(f"  Relations: {len(result['relations'])}")
    for rel in result['relations'][:3]:  # Show first 3 relations per chunk
        print(f"    • {rel}")
    if len(result['relations']) > 3:
        print(f"    ... and {len(result['relations']) - 3} more")

print("\n✅ Processing complete!")