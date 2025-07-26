import pandas as pd
import csv  # Still good to keep if quoting is used later

#  Step 1: Load PMIDs from .txt file into a DataFrame
with open("C:/Users/EZB-VM-06/Downloads/responce_pmids_output.txt", "r") as f:
    pmid_list = [line.strip() for line in f if line.strip()]  # Strip whitespace and ignore blank lines

first_df = pd.DataFrame({'Pubmed_ID': pmid_list})  # Create a DataFrame with column 'Pubmed_ID'

#  Step 2: Load the second CSV file (Obesity Missing PMIDs)
second_df = pd.read_csv("C:/Users/EZB-VM-06/Downloads/Obesity Missing PMIDs.csv", encoding='latin1')

#  Step 3: Clean column names (strip whitespace just in case)
second_df.columns = second_df.columns.str.strip()

#  Step 4: Ensure both columns are strings for accurate matching
first_df["Pubmed_ID"] = first_df["Pubmed_ID"].astype(str)
second_df["Missing PMIDs"] = second_df["Missing PMIDs"].astype(str)

#  Step 5: Add new column for matched PMIDs
second_df["Matched PMID"] = second_df["Missing PMIDs"].where(
    second_df["Missing PMIDs"].isin(first_df["Pubmed_ID"])
)

#  Step 6: Save result to a new CSV
second_df.to_csv("C:/Users/EZB-VM-06/Downloads/gene_responce_pmids_output_only_obesity.csv", index=False)

#  Optional print preview
print(" 'Matched PMID' column added to second CSV and saved.")
print(f"PMID List from TXT (first_df) head:\n{first_df.head()}")
print(f"\nSecond DataFrame (updated) head:\n{second_df.head()}")






## below code for csv 

# import pandas as pd
# import csv # Keep this import, though not directly used in this snippet, it's good practice if you might use csv.QUOTE_NONE etc.

# # Step 1: Load first Excel file with necessary columns
# # Use pd.read_excel for Excel files (.xlsx, .xls)
# # Remove encoding, quoting, and on_bad_lines as they are CSV-specific
# first_df = pd.read_excel(
#     "C:/Users/EZB-VM-06/Downloads/rkumar_output_file.xlsx", # Changed file extension to .xlsx (or .xls)
#     usecols=["Pubmed_ID", "Source", "Destination", "OriginalName"]
# )

# # Step 2: Load second CSV (this remains a CSV as per your request)
# second_df = pd.read_csv("C:/Users/EZB-VM-06/Downloads/Obesity Missing PMIDs.csv", encoding='latin1')

# # Step 3: Clean column names (strip whitespace)
# first_df.columns = first_df.columns.str.strip()
# second_df.columns = second_df.columns.str.strip()

# # Step 4: Convert ID columns to string for accurate comparison
# # This is crucial for matching, as Excel might read IDs as numbers
# first_df["Pubmed_ID"] = first_df["Pubmed_ID"].astype(str)
# second_df["Missing PMIDs"] = second_df["Missing PMIDs"].astype(str)

# # Step 5: Create a new column 'Matched PMID' in second_df
# # Fill it with 'Missing PMIDs' where a match exists in 'first_df["Pubmed_ID"]', else NaN
# second_df["Matched PMID"] = second_df["Missing PMIDs"].where(
#     second_df["Missing PMIDs"].isin(first_df["Pubmed_ID"])
# )

# # Step 6: Save the result (second_df to CSV)
# second_df.to_csv("C:/Users/EZB-VM-06/Downloads/New_updated_second_csv_with_matched_pmids.csv", index=False)

# print("✅ 'Matched PMID' column added to second CSV and saved.")
# print(f"First DataFrame (from Excel) head:\n{first_df.head()}")
# print(f"\nSecond DataFrame (updated) head:\n{second_df.head()}")