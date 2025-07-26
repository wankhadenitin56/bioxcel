import pandas as pd

# Load data
df_drugs = pd.read_excel('C:/Users/EZB-VM-06/Downloads/drug_class_list__25062025.xlsx')  # Main drug data
df_mapping = pd.read_excel('C:/Users/EZB-VM-06/Downloads/Drug Class List_2025-06-27_v2.xlsx')  # Class-Subclass mapping

# Convert mapping columns to lowercase
df_mapping['Subclass'] = df_mapping['Subclass'].str.lower().str.strip()
df_mapping['Class'] = df_mapping['Class'].str.lower().str.strip()

# Create mapping dictionary {Subclass → Class}
class_mapping = dict(zip(df_mapping['Subclass'], df_mapping['Class']))

# Function to map drug classes
def map_drug_class(drug_class):
    if pd.isna(drug_class) or drug_class == "":
        return None
    
    # Split into subclasses, strip whitespace, convert to lowercase
    subclasses = [s.strip().lower() for s in str(drug_class).split(';')]
    
    # Map each subclass to Class, keeping unique classes
    mapped_classes = list(set(
        class_mapping.get(subclass, " ") 
        for subclass in subclasses
    ))
    
    return "; ".join(mapped_classes)  # Combine unique classes

# Convert all columns to lowercase where possible
for col in df_drugs.columns:
    if df_drugs[col].dtype == object:
        df_drugs[col] = df_drugs[col].str.lower().str.strip()

# Apply mapping function
df_drugs['Class'] = df_drugs['Drug_Class'].apply(map_drug_class)

# Find duplicate rows (excluding the first occurrence)
duplicate_rows = df_drugs[df_drugs.duplicated(keep='first')]

# Print row numbers of duplicates (adding 2 to match Excel's row number, accounting for header)
if not duplicate_rows.empty:
    row_numbers = (duplicate_rows.index + 2).to_list()
    print(f"\nTotal duplicate rows found (excluding first occurrences): {len(duplicate_rows)}")
    print(f"Duplicate rows are at the following Excel row numbers:\n{row_numbers}")
else:
    print("\nNo duplicate rows found.")

# Save duplicates separately
duplicate_rows.to_excel('C:/Users/EZB-VM-06/Downloads/duplicate_rows_drugs.xlsx', index=False)

# Keep only first occurrence of duplicates
df_drugs = df_drugs.drop_duplicates(keep='first')

# Save cleaned data
df_drugs.to_excel('C:/Users/EZB-VM-06/Downloads/cleaned_drug_classes_removing_duplicates.xlsx', index=False)

print("\nDone! All data converted to lowercase, classes mapped, duplicates removed (keeping first occurrence), and duplicates saved separately.")
