# 🧬 Bioxcel Code Repository

This project is a collection of Python scripts and data files designed for biomedical NLP and data processing workflows. It includes functions for UMLS (Unified Medical Language System) concept matching, entity extraction, PubMed ID correlation, and machine learning-based molecule mapping.

---

## 📂 Project Structure

| File / Folder                  | Description                                                                 |
|-------------------------------|-----------------------------------------------------------------------------|
| `ES_match.py`                 | Script for fuzzy entity string matching using Elasticsearch or string sim. |
| `UMLS_All_Candidates.csv`     | Candidate UMLS terms extracted from input queries.                          |
| `UMLS_Match_Results_*.csv`    | Results of various UMLS concept matching approaches.                        |
| `check_csv_encoding.py`       | Script to check encoding of CSV files.                                     |
| `comparison.py`               | Code for comparing similarity between biomedical terms.                    |
| `concepi_id_search.py`        | Utility for searching Concept IDs.                                         |
| `concept_id_xyz.py`           | Concept ID mapping and testing module.                                     |
| `data.py`                     | Data loading or utility functions.                                         |
| `database.py`                 | Code for database interactions (likely SQLite or MySQL).                   |
| `genai.py`                    | Code for integrating Generative AI tools or models.                        |
| `gliner_model_news.py`        | Named Entity Recognition (NER) with GLiNER on biomedical news.             |
| `llama3.py`                   | LLaMA3 model interface or prompt template integration.                     |
| `map_functional_molecule.py` | Mapping molecules to their function based on NLP output.                   |
| `news_*.py`                   | Multiple scripts for biomedical news analysis and correlation.             |
| `pkcb.py`                     | Likely a processing module for kinase or bio-entity correlations.          |
| `pmid_match.py`               | Matching PubMed IDs with biomedical entities or terms.                     |
| `umls_match_score.py`         | Scoring or ranking UMLS matches.                                           |
| `web_scrapping.py`            | Script for scraping biomedical web data.                                   |
| `xyz.py`, `test.py`           | Miscellaneous or utility/testing modules.                                  |

---

## ⚙️ Technologies Used

- **Python 3**
- **Pandas**
- **scikit-learn**
- **Elasticsearch / difflib / fuzzywuzzy**
- **UMLS / PubMed**
- **Generative AI (OpenAI, LLaMA3)**

---

## 🚀 Features

- UMLS Concept Matching & Scoring  
- PubMed Article and Entity Mapping  
- Named Entity Recognition with GLiNER  
- Biomedical News Processing and Correlation  
- Web Scraping of Bioinformatic Content  
- LLM integration for entity completion / prompts

---

## 🧪 Getting Started

1. Clone the repository:

```bash
git clone https://github.com/bioxcelcode/bioxcel.git
cd bioxcel
