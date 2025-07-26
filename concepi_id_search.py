import pandas as pd
import chardet # Used for encoding detection

def interactive_concept_search(dataframe):
    
    # Ensure 'Concept_ID' column exists and is string type
    if 'concept_id' not in dataframe.columns:
        print("Error: 'Concept_ID' column not found in the DataFrame.")
        print(f"Available columns are: {dataframe.columns.tolist()}")
        return

    # Clean the 'Concept_ID' column by converting to string and stripping whitespace
    dataframe['concept_id'] = dataframe['concept_id'].astype(str).str.strip()

    print("\n--- Interactive Concept ID Search ---")
    print("Enter 'exit' or 'quit' to stop the search.")

    while True:
        user_input_id = input("\nEnter the Concept_ID to search: ").strip()

        if user_input_id.lower() in ['exit', 'quit']:
            print("Exiting search. Goodbye!")
            break

        if not user_input_id:
            print("Input cannot be empty. Please enter a Concept_ID.")
            continue

        # Filter the DataFrame for exact matches
        results = dataframe[dataframe['concept_id'] == user_input_id]

        if not results.empty:
            print(f"\n--- Results for concept_id: '{user_input_id}' ---")
            # Print all columns for the matching rows without the DataFrame index
            print(results.to_string(index=False))
            print("------------------------------------------")
        else:
            print(f"No matching rows found for Concept_ID: '{user_input_id}'")

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    
    csv_file_path = "C:/Users/EZB-VM-06/Downloads/Concept_ID_Map_ALL_Types (2).csv"

    my_dataframe = pd.DataFrame() # Initialize as empty DataFrame

    # --- Load your actual CSV here with robust encoding handling ---
    try:
        # Function to detect encoding 
        def detect_file_encoding(filepath):
            with open(filepath, 'rb') as f:
                rawdata = f.read(100000) # Read larger sample for better detection
            result = chardet.detect(rawdata)
            return result['encoding']

        detected_encoding = detect_file_encoding(csv_file_path)
        print(f"Detected encoding by chardet: {detected_encoding}")

        # Define a list of common encodings to try, in order of likelihood
        # Start with detected, then utf-8, then common Windows/Western encodings
        encodings_to_try = [detected_encoding, 'utf-8', 'cp1252', 'latin-1', 'ISO-8859-1']
        
        # Remove duplicates and None from the list, prioritizing earlier entries
        encodings_to_try = list(pd.unique([e for e in encodings_to_try if e is not None]))

        file_loaded = False
        for encoding in encodings_to_try:
            print(f"Attempting to load with encoding: '{encoding}'...")
            try:
                my_dataframe = pd.read_csv(csv_file_path, encoding=encoding, on_bad_lines='skip', engine='python')
                file_loaded = True
                print(f"Successfully loaded with encoding: '{encoding}'")
                break # Exit loop if loading is successful
            except UnicodeDecodeError as e:
                print(f"  UnicodeDecodeError with '{encoding}': {e}")
                continue # Try next encoding
            except Exception as e:
                print(f"  Other error with '{encoding}': {e}")
                continue # Try next encoding

        if not file_loaded:
            raise Exception("Failed to load the CSV file with all attempted encodings.")

        # Clean column names (strip whitespace from column headers)
        my_dataframe.columns = my_dataframe.columns.str.strip()

        print(f"\nCSV file '{csv_file_path}' loaded successfully with {len(my_dataframe)} rows.")
        print(f"Columns found in file: {my_dataframe.columns.tolist()}")

        # Call the interactive search function if the DataFrame is not empty
        if not my_dataframe.empty:
            interactive_concept_search(my_dataframe)
        else:
            print("DataFrame is empty after loading. Please check the CSV file content or its encoding.")

    except FileNotFoundError:
        print(f"Error: The file '{csv_file_path}' was not found. Please double-check the path.")
    except Exception as e:
        print(f"An unexpected error occurred during file loading or processing: {e}")


# from transformers import pipeline

# company_ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")

# text ="""Tyra Biosciences Announces Preclinical Results for FGFR Inhibitor"""
# entities = company_ner(text)
# for ent in entities:
#     if ent["entity_group"] == "ORG":
#         print(ent["word"])
