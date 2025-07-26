import pandas as pd

# Load cleaned drug file
df_drugs = pd.read_excel('C:/Users/EZB-VM-06/Downloads/cleaned_drug_classes_removing_duplicates.xlsx')
df_drugs['Drug'] = df_drugs['Drug'].str.lower().str.strip()

# Load PubMed file
df_pubmed = pd.read_excel('C:/Users/EZB-VM-06/Downloads/last_10_year.xlsx')  ## nwankhade@bioxcel.com_output_file (1)
df_pubmed['OriginalName'] = df_pubmed['OriginalName'].str.lower().str.strip()

# Keep only relevant columns from PubMed file
pubmed_columns_to_keep = [
    'Pubmed_ID', 'Title', 'Abstract', 'Source', 'Destination', 'OriginalName',
    'UmlsConfScore', 'Concept_ID', 'DestinationSemanticType', 'FinalType', 'FinalName'
]
df_pubmed_filtered = df_pubmed[pubmed_columns_to_keep]

# Select relevant columns from drug file
drug_columns_to_merge = ['Drug','Drug_Class', 'Class']       # 'MoA', 'Originator_All', 'Developer_All', 'Phase', 'Disease_SplitResultList',

# Merge PubMed with drug file
df_merged = df_pubmed_filtered.merge(df_drugs[drug_columns_to_merge],
                                     left_on='OriginalName',
                                     right_on='Drug',
                                     how='left')

# Split matched and unmatched rows
df_matched = df_merged[df_merged['Drug'].notna()]    # Rows where match happened
df_unmatched = df_merged[df_merged['Drug'].isna()]   # Rows where no match happened

# Save matched rows
df_matched.to_excel('C:/Users/EZB-VM-06/Downloads/pubmed_with_drug_class_matched_last10YEAR.xlsx', index=False)

# Save unmatched rows separately
#df_unmatched.to_excel('C:/Users/EZB-VM-06/Downloads/pubmed_no_drug_match.xlsx', index=False)

print(f"\nTotal matched rows: {len(df_matched)}")
print(f"Total unmatched rows: {len(df_unmatched)}")
print("\nFiles saved successfully.")




# import pandas as pd

# # Load cleaned drug file
# df_drugs = pd.read_excel('C:/Users/EZB-VM-06/Downloads/cleaned_drug_classes_removing_duplicates.xlsx')

# # Convert Drug column to lowercase and strip whitespaces
# df_drugs['Drug'] = df_drugs['Drug'].str.lower().str.strip()

# # Load second Excel file (PubMed-related)
# df_pubmed = pd.read_excel('C:/Users/EZB-VM-06/Downloads/nwankhade@bioxcel.com_output_file (1).xlsx')  # <-- Change filename if needed

# # Convert OriginalName to lowercase and strip whitespaces
# df_pubmed['OriginalName'] = df_pubmed['OriginalName'].str.lower().str.strip()

# # Merge on OriginalName from df_pubmed with Drug from df_drugs
# df_merged = df_pubmed.merge(df_drugs[['Drug', 'Drug_Class', 'Class']], 
#                             left_on='OriginalName', 
#                             right_on='Drug', 
#                             how='left')

# # Save merged result
# df_merged.to_excel('C:/Users/EZB-VM-06/Downloads/pubmed_with_drug_class.xlsx', index=False)

# print("\nDone! Merged PubMed file with Drug_Class and Class based on OriginalName.")


