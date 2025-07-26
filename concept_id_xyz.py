import pandas as pd

# Read the CSV file
df = pd.read_csv("C:/Users/EZB-VM-06/Downloads/Concept_ID_Map_ALL_Types (1).csv",encoding='latin1')

# List of concept IDs to search for
concept_ids_to_search = [
    "C2261576", "C1172412", "C1706144", "C0250416",
    "C1259877", "C1335261", "C1418913", "C1565919"
]

# Filter rows where 'concept_id' matches any in the list
filtered_df = df[df['concept_id'].isin(concept_ids_to_search)]

# Show the filtered rows
print(filtered_df)

# Optional: Save the filtered result to a new CSV file
filtered_df.to_csv("C:/Users/EZB-VM-06/Downloads/filtered_concepts.csv", index=False)
