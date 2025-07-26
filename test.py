import pandas as pd
import spacy
import scispacy
from collections import defaultdict
import pycountry
import csv
from io import StringIO
import chardet
import re
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import torch

class EnhancedBiomedicalExtractor:
    def __init__(self):
        """Initialize multiple models for comprehensive entity extraction"""
        self.models = {}
        self.load_models()
    
    def load_models(self):
        """Load various NLP models for entity extraction"""
        print("Loading models...")
        
        # Load spaCy models
        try:
            self.models['general'] = spacy.load("en_core_web_sm")
            print("✓ General spaCy model loaded")
        except:
            print("✗ General spaCy model failed to load")
        
        # Try multiple biomedical spaCy models
        biomedical_models = [
            "en_ner_bc5cdr_md",
            "en_ner_bionlp13cg_md", 
            "en_core_sci_sm"
        ]
        
        for model_name in biomedical_models:
            try:
                self.models[f'bio_{model_name}'] = spacy.load(model_name)
                print(f"✓ {model_name} loaded")
                break
            except:
                print(f"✗ {model_name} not available")
        
        # Load transformer-based biomedical models
        try:
            # BioBERT for biomedical NER
            self.models['biobert'] = pipeline(
                "ner",
                model="dmis-lab/biobert-base-cased-v1.1-squad",
                tokenizer="dmis-lab/biobert-base-cased-v1.1-squad",
                aggregation_strategy="simple"
            )
            print("✓ BioBERT NER model loaded")
        except Exception as e:
            print(f"✗ BioBERT model failed: {e}")
        
        # Load PubMedBERT for better biomedical understanding
        try:
            self.models['pubmedbert'] = pipeline(
                "ner",
                model="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
                aggregation_strategy="simple"
            )
            print("✓ PubMedBERT loaded")
        except Exception as e:
            print(f"✗ PubMedBERT failed: {e}")
    
    def extract_with_regex_patterns(self, text):
        """Extract entities using regex patterns as fallback"""
        entities = defaultdict(list)
        
        # Drug name patterns (common suffixes and prefixes)
        drug_patterns = [
            r'\b\w*(?:cillin|mycin|oxin|azole|prazole|sartan|statin|mab|nib|tidine)\b',
            r'\b(?:anti|meta|para|ortho|proto)-\w+\b',
            r'\b\w+(?:ol|al|ine|ate|ide|ium)\b(?=\s+(?:mg|mcg|g|ml|tablet|capsule|injection))',
        ]
        
        # Disease patterns
        disease_patterns = [
            r'\b\w*(?:itis|osis|emia|pathy|trophy|plasia|carcinoma|sarcoma|lymphoma)\b',
            r'\b(?:syndrome|disease|disorder|condition|infection|tumor|cancer)\b',
            r'\b\w+\s+(?:syndrome|disease|disorder|condition|infection|tumor|cancer)\b',
        ]
        
        for pattern in drug_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities['drugs'].extend([m.strip() for m in matches if len(m.strip()) > 2])
        
        for pattern in disease_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities['diseases'].extend([m.strip() for m in matches if len(m.strip()) > 3])
        
        return entities
    
    def extract_entities_comprehensive(self, text):
        """Extract entities using multiple models and approaches"""
        entities = defaultdict(list)
        
        if not text or pd.isna(text):
            return entities
        
        text = str(text).strip()
        if not text:
            return entities
        
        print(f"\n--- Processing: '{text[:100]}...' ---")
        
        # Method 1: General spaCy model for countries and organizations
        if 'general' in self.models:
            doc_gen = self.models['general'](text)
            print("  General Model Entities:")
            for ent in doc_gen.ents:
                print(f"    - '{ent.text}' ({ent.label_})")
                if ent.label_ == "GPE":
                    try:
                        if pycountry.countries.search_fuzzy(ent.text):
                            entities['countries'].append(ent.text)
                    except LookupError:
                        pass
                elif ent.label_ == "ORG":
                    entities['companies'].append(ent.text)
        
        # Method 2: Biomedical spaCy models
        for model_key in self.models:
            if model_key.startswith('bio_'):
                try:
                    doc_bio = self.models[model_key](text)
                    print(f"  {model_key} Entities:")
                    for ent in doc_bio.ents:
                        print(f"    - '{ent.text}' ({ent.label_})")
                        if ent.label_ in ["DISEASE", "DISORDER"]:
                            entities['diseases'].append(ent.text)
                        elif ent.label_ in ["CHEMICAL", "DRUG"]:
                            entities['drugs'].append(ent.text)
                except Exception as e:
                    print(f"    Error with {model_key}: {e}")
        
        # Method 3: Transformer-based models
        if 'biobert' in self.models:
            try:
                bert_results = self.models['biobert'](text)
                print("  BioBERT Entities:")
                for ent in bert_results:
                    print(f"    - '{ent['word']}' ({ent['entity_group']}) - {ent['score']:.3f}")
                    if ent['score'] > 0.5:  # Confidence threshold
                        if 'DISEASE' in ent['entity_group'].upper():
                            entities['diseases'].append(ent['word'])
                        elif 'CHEMICAL' in ent['entity_group'].upper() or 'DRUG' in ent['entity_group'].upper():
                            entities['drugs'].append(ent['word'])
            except Exception as e:
                print(f"    BioBERT error: {e}")
        
        # Method 4: Regex patterns as fallback
        regex_entities = self.extract_with_regex_patterns(text)
        for key, values in regex_entities.items():
            entities[key].extend(values)
        
        # Clean and deduplicate
        for key in entities:
            entities[key] = list(set([item.strip() for item in entities[key] if item.strip()]))
        
        return entities

def process_dataframe_enhanced(df):
    """Process dataframe with enhanced entity extraction"""
    extractor = EnhancedBiomedicalExtractor()
    
    # Auto-detect text columns
    text_cols = []
    for col in df.columns:
        if df[col].dtype == 'object':
            # Check if column contains substantial text
            sample_text = df[col].dropna().head(5).astype(str).str.len().mean()
            if sample_text > 20:  # Columns with average length > 20 chars
                text_cols.append(col)
    
    print(f"Auto-detected text columns: {text_cols}")
    
    # Initialize output columns
    for col in ['countries', 'diseases', 'drugs', 'companies']:
        df[col] = None
    
    print(f"Processing {len(df)} rows...")
    
    for idx, row in df.iterrows():
        if idx % 10 == 0:
            print(f"Processing row {idx}/{len(df)}...")
        
        try:
            # Combine text from detected columns
            combined_text = " ".join(str(row[col]) for col in text_cols if pd.notna(row[col]))
            if not combined_text.strip():
                continue
            
            entities = extractor.extract_entities_comprehensive(combined_text)
            
            for key in ['countries', 'diseases', 'drugs', 'companies']:
                if entities.get(key):
                    df.at[idx, key] = entities[key]
        
        except Exception as e:
            print(f"Error at row {idx}: {e}")
    
    return df

def safe_read_csv(filepath):
    """Read CSV with encoding detection and fallback strategies"""
    def detect_encoding(filepath):
        with open(filepath, 'rb') as f:
            rawdata = f.read(10000)
        return chardet.detect(rawdata)['encoding']
    
    encoding = detect_encoding(filepath)
    print(f"Detected encoding: {encoding}")
    
    try:
        return pd.read_csv(filepath, encoding=encoding, on_bad_lines='skip')
    except Exception as e:
        print(f"Failed with detected encoding: {e}")
        try:
            return pd.read_csv(filepath, encoding='utf-8', on_bad_lines='skip')
        except Exception as e2:
            print(f"Fallback UTF-8 read failed: {e2}")
            raise

def save_results_safely(df, output_path):
    """Save results safely to a CSV"""
    df_copy = df.copy()
    for col in ['countries', 'diseases', 'drugs', 'companies']:
        if col in df_copy.columns:
            df_copy[col] = df_copy[col].apply(
                lambda x: ', '.join(x) if isinstance(x, list) else str(x) if x else ''
            )
    try:
        df_copy.to_csv(output_path, index=False, encoding='utf-8')
        print(f"Results saved to {output_path}")
    except Exception as e:
        print(f"UTF-8 save failed: {e}")
        df_copy.to_csv(output_path, index=False, encoding='latin-1')
        print(f"Saved with latin-1 encoding instead.")

# === MAIN EXECUTION ===
if __name__ == "__main__":
    input_file = 'C:/Users/EZB-VM-06/Downloads/daily_news_report_2025-07-10_270_records.csv'
    output_file = 'C:/Users/EZB-VM-06/Downloads/enhanced_entity_extraction_results.csv'
    
    print("Reading file...")
    df = safe_read_csv(input_file)
    
    print("Extracting entities with enhanced models...")
    processed_df = process_dataframe_enhanced(df)
    
    print("Saving results...")
    save_results_safely(processed_df, output_file)
    
    print("\nSummary:")
    for col in ['countries', 'diseases', 'drugs', 'companies']:
        if col in processed_df.columns:
            non_empty = processed_df[col].notna().sum()
            print(f"{col}: {non_empty} rows with entities")
    
    print("\nEntity extraction complete!")
    
    # Show sample results
    print("\nSample Results:")
    for col in ['countries', 'diseases', 'drugs', 'companies']:
        if col in processed_df.columns:
            sample_entities = processed_df[col].dropna().head(3).tolist()
            if sample_entities:
                print(f"{col}: {sample_entities}")