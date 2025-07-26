# import pandas as pd
# import re
# from gliner import GLiNER
# from transformers import AutoTokenizer
# from langdetect.lang_detect_exception import LangDetectException

# #  Chunking function to avoid truncation
# def chunk_text(text, chunk_size=384, overlap=20):   #384
#     sentences = re.split(r'(?<=[.!?]) +', text)
#     chunks = []
#     current_chunk = ""

#     for sentence in sentences:
#         if len(current_chunk) + len(sentence) <= chunk_size:
#             current_chunk += sentence + " "
#         else:
#             chunks.append(current_chunk.strip())
#             current_chunk = sentence + " "

#     if current_chunk:
#         chunks.append(current_chunk.strip())

#     if overlap > 0 and len(chunks) > 1:
#         overlapped_chunks = []
#         for i in range(len(chunks)):
#             current = chunks[i]
#             if i > 0:
#                 prev = chunks[i - 1].split()[-overlap:]
#                 current = " ".join(prev) + " " + current
#             overlapped_chunks.append(current.strip())
#         return overlapped_chunks

#     return chunks

# #  Load GLiNER model
# gliner_model = GLiNER.from_pretrained("urchade/gliner_largev2") ##  knowledgator/gliner-x-large  urchade/gliner_largev2
# custom_labels = ["Drug", "Disease", "Organization", "Country"]

# #  Load your CSV file
# df = pd.read_csv("C:/Users/EZB-VM-06/Downloads/news__newss__202507161357.csv")  

# #  Prepare output columns
# df["drugs"] = ""
# df["diseases"] = ""
# df["organizations"] = ""
# df["countries"] = ""

# #  Process each row
# for idx, row in df.iterrows():
#     title = str(row.get("title", ""))
#     description = str(row.get("description", ""))
#     full_text = f"{title}\n{description}"
#     chunks = chunk_text(full_text, chunk_size=384)

#     all_entities = []
#     for chunk in chunks:
#         try:
#             entities = gliner_model.predict_entities(
#                 chunk, custom_labels, threshold=0.4, flat_ner=True
#             )
#             all_entities.extend(entities)
#         except Exception as e:
#             print(f" Error in row {idx}: {e}")

#     # Organize entities by type
#     drug_list = sorted(set(ent["text"] for ent in all_entities if ent["label"] == "Drug"))
#     disease_list = sorted(set(ent["text"] for ent in all_entities if ent["label"] == "Disease"))
#     org_list = sorted(set(ent["text"] for ent in all_entities if ent["label"] == "Organization"))
#     country_list = sorted(set(ent["text"] for ent in all_entities if ent["label"] == "Country"))

#     # Save to new columns
#     df.at[idx, "drugs"] = ", ".join(drug_list)
#     df.at[idx, "diseases"] = ", ".join(disease_list)
#     df.at[idx, "organizations"] = ", ".join(org_list)
#     df.at[idx, "countries"] = ", ".join(country_list)

#     print(f" Processed row {idx + 1}/{len(df)}")

# #  Save final output
# df.to_excel("C:/Users/EZB-VM-06/Downloads/gliner_extracted_output_2.xlsx", index=False)
# print("\n Extraction complete! Saved to 'gliner_extracted_output_2.xlsx'")

#####################################  working code  #########################################################

import pandas as pd
import re
from gliner import GLiNER
from transformers import AutoTokenizer
import time
#  Load GLiNER model and tokenizer
gliner_model = GLiNER.from_pretrained("knowledgator/gliner-x-large")
tokenizer = AutoTokenizer.from_pretrained("knowledgator/gliner-x-large")

# gliner_model = GLiNER.from_pretrained("urchade/gliner_multi-v2.1")
# tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")


custom_labels = ["Drug", "Disease", "Organization", "Country"]

#  Token-aware chunking function to avoid truncation
def chunk_text_by_tokens(text, tokenizer, max_tokens=512, overlap=20):   ## 384 for x xlm-roberta-large model 
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = len(words)
        for i in range(start + max_tokens, start, -1):
            chunk = " ".join(words[start:i])
            tokenized = tokenizer(chunk, truncation=False, return_tensors="pt")
            if tokenized.input_ids.shape[1] <= max_tokens:
                end = i
                break
        chunks.append(" ".join(words[start:end]))
        start = end - overlap if end - overlap > start else end

    return chunks

#  Load your CSV file
df = pd.read_csv("C:/Users/EZB-VM-06/Downloads/news__newss__202507161357.csv") #New_one_month__newss__202507161841
# df = pd.read_csv("C:/Users/EZB-VM-06/Downloads/daily_news_report_2025-07-09_402_records (1).csv") #New_one_month__newss__202507161841


#  Prepare output columns
df["drugs"] = ""
df["diseases"] = ""
df["organizations"] = ""
df["countries"] = ""

#  Process each row
start_time = time.time()
for idx, row in df.iterrows():
    title = str(row.get("title", ""))
    description = str(row.get("description", ""))

    # title = str(row.get("Title", ""))
    # description = str(row.get("Description", ""))

    full_text = f"{title}\n{description}"

    #  Token-aware chunking
    chunks = chunk_text_by_tokens(full_text, tokenizer, max_tokens=512, overlap=20)

    all_entities = []
    for chunk in chunks:
        try:
            entities = gliner_model.predict_entities(
                chunk,
                custom_labels,
                threshold=0.70,  # TH = 70 is working well only 5% drugs miss to capture
                flat_ner=True
            )
            all_entities.extend(entities)
        except Exception as e:
            print(f" Error in row {idx}: {e}")

    #  Organize and save entities
    drug_list = sorted(set(ent["text"] for ent in all_entities if ent["label"] == "Drug"))
    disease_list = sorted(set(ent["text"] for ent in all_entities if ent["label"] == "Disease"))
    org_list = sorted(set(ent["text"] for ent in all_entities if ent["label"] == "Organization"))
    country_list = sorted(set(ent["text"] for ent in all_entities if ent["label"] == "Country"))

    df.at[idx, "drugs"] = ", ".join(drug_list)
    df.at[idx, "diseases"] = ", ".join(disease_list)
    df.at[idx, "organizations"] = ", ".join(org_list)
    df.at[idx, "countries"] = ", ".join(country_list)

    print(f" Processed row {idx + 1}/{len(df)}")

#  Save final output
output_path = "C:/Users/EZB-VM-06/Downloads/071_gliner_largeModel_1028.xlsx"
df.to_excel(output_path, index=False)
print(f"\n Extraction complete! Saved to '{output_path}'")
print(" Total time:", time.time() - start_time, "seconds")






############ below code is used with batch processing and above without batch processing #################

# import pandas as pd
# import re
# import time
# from gliner import GLiNER
# from transformers import AutoTokenizer

# # Load GLiNER model and tokenizer
# gliner_model = GLiNER.from_pretrained("knowledgator/gliner-x-large")
# tokenizer = AutoTokenizer.from_pretrained("knowledgator/gliner-x-large")

# custom_labels = ["Drug", "Disease", "Organization", "Country"]

# # Token-aware chunking function to avoid truncation
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

# # Load CSV
# df = pd.read_csv("C:/Users/EZB-VM-06/Downloads/New_one_month__newss__202507161841.csv")
# df["drugs"] = ""
# df["diseases"] = ""
# df["organizations"] = ""
# df["countries"] = ""

# # Prepare full_text per row
# all_chunks = []
# chunk_map = []  # Keeps (row_idx, chunk_text)

# for idx, row in df.iterrows():
#     title = str(row.get("title", ""))
#     description = str(row.get("description", ""))
#     full_text = f"{title}\n{description}"

#     chunks = chunk_text_by_tokens(full_text, tokenizer, max_tokens=512, overlap=20)
#     for chunk in chunks:
#         all_chunks.append(chunk)
#         chunk_map.append(idx)  # Map this chunk to the original row

# print(f"Total chunks to process: {len(all_chunks)}")

# # Batch prediction
# start_time = time.time()
# batch_size = 32
# all_entities_per_chunk = []

# for i in range(len(all_chunks)):
#     try:
#         entities = gliner_model.predict_entities(
#             all_chunks[i],
#             custom_labels,
#             threshold=0.75,
#             flat_ner=True
#         )
#         all_entities_per_chunk.append(entities)
#     except Exception as e:
#         print(f"Error in chunk {i}: {e}")
#         all_entities_per_chunk.append([])


# # Organize entities back to rows
# row_entities_map = {idx: [] for idx in df.index}

# for chunk_entities, row_idx in zip(all_entities_per_chunk, chunk_map):
#     row_entities_map[row_idx].extend(chunk_entities)

# for idx in df.index:
#     ents = row_entities_map[idx]
#     df.at[idx, "drugs"] = ", ".join(sorted(set(ent["text"] for ent in ents if ent["label"] == "Drug")))
#     df.at[idx, "diseases"] = ", ".join(sorted(set(ent["text"] for ent in ents if ent["label"] == "Disease")))
#     df.at[idx, "organizations"] = ", ".join(sorted(set(ent["text"] for ent in ents if ent["label"] == "Organization")))
#     df.at[idx, "countries"] = ", ".join(sorted(set(ent["text"] for ent in ents if ent["label"] == "Country")))

#     if idx % 25 == 0 or idx == len(df) - 1:
#         print(f" Processed row {idx + 1}/{len(df)}")

# # Save output
# output_path = "C:/Users/EZB-VM-06/Downloads/gliner_extracted_output_batch.xlsx"
# df.to_excel(output_path, index=False)
# print(f"\n Extraction complete! Saved to '{output_path}'")
# print("Total time:", round(time.time() - start_time, 2), "seconds")









# import random
# from gliner import GLiNER

# # Load GLiNER model
# gliner_model = GLiNER.from_pretrained("urchade/gliner_largev2")

# # --- INPUT ---
# title = "Pfizer announces positive results for their RSV vaccine"
# description = """
# The GPRC5D-directed therapies market is witnessing robust growth, driven by the high unmet need in relapsed/refractory multiple myeloma. Innovative modalities like CAR-T cells, bispecific antibodies, and antibody-drug conjugates targeting GPRC5D are advancing rapidly through clinical pipelines. With limited cross-reactivity to normal tissues, GPRC5D presents a promising target, enhancing its commercial potential.
# LAS VEGAS , July 3, 2025 /PRNewswire/ -- DelveInsight's GPRC5D-directed Therapies Market Size, Target Population, Competitive Landscape & Market Forecast report includes a comprehensive understanding of current treatment practices, emerging GPRC5D-directed therapies, market share of individual therapies, and current and forecasted GPRC5D-directed therapies market size from 2020 to 2034, segmented into 7MM [ the United States , the EU4 ( Germany , France , Italy , and Spain ), the United Kingdom , and Japan ].
# Key Takeaways from the GPRC5D-directed Therapies Market Report
# As per DelveInsight's analysis, the total market size of GPRC5D-directed therapies in the 7MM is expected to surge significantly by 2034. As per DelveInsight's estimates, the total market size in the US for multiple myeloma was ~USD 15 billion in 2024, which is expected to show positive growth. Currently, TALVEY (talquetamab-tgvs) , approved in August 2023 , is the first and the only approved GPRC5D-directed therapy in the market. In the US, there were around 33,700 incident cases of multiple myeloma in 2024, which was the highest among seven major markets. Leading GPRC5D-directed therapies companies, such as Bristol Myers Squibb (Juno Therapeutics), AstraZeneca, Oricell, CARsgen Therapeutics, Sanofi, Jiangsu Simcere Pharmaceutical, AbbVie, and others, are developing novel GPRC5D-directed therapies that can be available in the GPRC5D-directed therapies market in the coming years. Some of the key GPRC5D-directed therapies in the pipeline include Arlocabtagene autoleucel (BMS-986393), AZD0305, OriCAR-017, CT071, SAR446523 , SIM0500, and others. In June 2025 , CARsgen Therapeutics presented a Phase I study of CT071 for the treatment of newly diagnosed multiple myeloma at the European Hematology Association (EHA) website. Discover which indication is expected to grab the major GPRC5D-directed therapies market share @ GPRC5D-directed Therapies Market Report
# GPRC5D-directed Therapies Market Dynamics
# The GPRC5D-directed therapies market is emerging as a significant focus within the landscape of multiple myeloma treatment. GPRC5D is an orphan receptor predominantly expressed on malignant plasma cells, with limited expression in normal tissues, making it an attractive therapeutic target. Its expression is independent of BCMA , allowing it to serve as a complementary or alternative target, particularly in patients who have relapsed after BCMA-directed treatments. This has driven a wave of innovation in bispecific antibodies, ADCs, and CAR-T cell therapies targeting GPRC5D .
# The competitive landscape is rapidly evolving, with multiple biotech and pharmaceutical companies advancing GPRC5D-targeted candidates through preclinical and clinical pipelines. Notably, companies such as Johnson & Johnson (via its teclistamab platform) and Legend Biotech are exploring GPRC5D-directed bispecifics and CAR-T constructs, respectively. Early-phase clinical trials have shown promising efficacy in relapsed/refractory multiple myeloma (RRMM) populations , even among those previously treated with BCMA-targeted agents. The ability of GPRC5D-targeting therapies to produce deep and durable responses , with manageable toxicity profiles, is a key factor fueling investor interest and partnerships in this space.
# Market dynamics are further shaped by the unmet clinical need in late-line multiple myeloma settings . As more patients receive BCMA therapies earlier in their treatment journeys, resistance and relapse rates have prompted the search for alternative targets like GPRC5D. Regulatory agencies , including the FDA and EMA, have shown openness to accelerated pathways for novel immunotherapies in RRMM, creating favorable conditions for GPRC5D-focused assets. Additionally, the ongoing development of allogeneic platforms and off-the-shelf bispecific antibodies is poised to enhance patient access and reduce logistical burdens compared to autologous CAR-T approaches.
# However, challenges remain in this emerging space. Differentiation among therapies targeting the same antigen will be crucial, as companies aim to optimize dosing schedules, minimize on-target off-tumor toxicities, and address manufacturing scalability. Pricing and reimbursement strategies will also play a critical role, particularly as competition intensifies and cost-effectiveness comes under scrutiny from payers and health systems.
# Looking forward, the GPRC5D-directed therapy market is expected to expand significantly over the next 10 years, driven by a combination of clinical validation, rising relapse rates in existing treatment paradigms, and an influx of new therapeutic modalities . Success in this domain will likely depend on the ability to integrate GPRC5D-targeted agents into combination regimens, leverage companion diagnostics to optimize patient selection, and establish long-term safety and efficacy through robust Phase III trials and real-world evidence.
# GPRC5D-directed Therapies Treatment Market
# GPRC5D is highly expressed in multiple myeloma cells and is found in abundance within the bone marrow of patients diagnosed with multiple myeloma or smoldering multiple myeloma. Currently, TALVEY (talquetamab-tgvs) , approved in August 2023 , is the first and the only approved GPRC5D-directed therapy in the market. TALVEY is a novel, first-in-class bispecific T-cell engager that simultaneously binds to the CD3 receptor on T cells and GPRC5D, which is present on multiple myeloma cells, non-cancerous plasma cells, and certain healthy epithelial tissues, particularly in the skin and tongue. It is administered subcutaneously on a weekly or biweekly basis following an initial step-up dosing phase, giving physicians the flexibility to tailor dosing schedules to individual patient needs.
# Learn more about the FDA-approved GPRC5D-directed therapies @ Approved GPRC5D-directed Therapies
# Key Emerging GPRC5D-directed Therapies and Companies
# Some of the major emerging key players in the GPRC5D market include CARsgen Therapeutics (CT071), Oricell (OriCAR-017), Bristol- Myers Squibb (Arlocabtagene autoleucel), AstraZeneca (AZD0305), and others.
# OriCAR-017 is an innovative CAR T cell therapy developed to target GPRC5D, aimed at treating relapsed or refractory multiple myeloma. Developed using Oricell Therapeutics' advanced proprietary technology, OriCAR-017 demonstrates unique advantages in terms of binding strength, cell persistence, tumor-fighting capability, and safety. It received Investigational New Drug (IND) approval from the U.S. FDA in January 2024 , following earlier approval by China's NMPA in 2023.
# BMS-986393 is an autologous CAR T cell therapy also directed at GPRC5D. Currently under evaluation in the Phase II QUINTESSENTIAL trial for relapsed/refractory multiple myeloma, the therapy is expected to yield key data in 2026. At the ASH 2023 conference, updated results from its Phase I trial highlighted durable responses, manageable safety, and clinical activity in patients regardless of prior BCMA-targeted treatment exposure.
# The anticipated launch of these emerging therapies are poised to transform the GPRC5D-directed therapies market landscape in the coming years. As these cutting-edge therapies continue to mature and gain regulatory approval, they are expected to reshape the GPRC5D-directed therapies market landscape, offering new standards of care and unlocking opportunities for medical innovation and economic growth.
# To know more about GPRC5D-directed therapies clinical trials, visit @ GPRC5D-directed Therapies Treatment
# GPRC5D-directed Therapies Overview
# GPRC5D, an orphan G-protein-coupled receptor located on chromosome 12p13, lacks a known ligand. It has emerged as a validated therapeutic target in multiple myeloma. Early-phase clinical trials involving T–cell–reducing therapies aimed at GPRC5D have demonstrated encouraging efficacy along with tolerable safety profiles, although these findings still require confirmation in Phase III studies.
# This receptor is highly expressed in multiple myeloma cells and is prevalent in the bone marrow of patients with both active and smoldering disease. Furthermore, its mRNA levels are notably higher in multiple myeloma compared to other hematologic malignancies. The selective overexpression of GPRC5D in myeloma cells makes it an attractive candidate for immune effector cell–based therapies. Targeting GPRC5D, either as a standalone treatment or in combination with other anti-myeloma agents, could broaden the therapeutic landscape, particularly for patients who require novel mechanisms of action, whether to delay the use of BCMA-targeted therapies, overcome poor responses or antigen loss, or address clonal variability.
# Although GPRC5D and BCMA are both expressed on CD138+ myeloma cells, their expression is not interdependent, making them distinct therapeutic targets. Importantly, GPRC5D expression remains stable even when BCMA is lost, a known resistance mechanism in some relapsed patients, supporting the rationale for combining GPRC5D- and BCMA-directed T-cell therapies to better manage the disease's heterogeneity.
# GPRC5D-directed Therapies Epidemiology Segmentation
# The GPRC5D-directed therapies market report proffers epidemiological analysis for the study period 2020–2034 in the 7MM, segmented into:
# Total Cases in Selected Indications (Multiple Myeloma) for GPRC5D-directed Therapies Total Eligible Patient Pool in Selected Indications for GPRC5D-directed Therapies Total Treated Cases in Selected Indications for GPRC5D-directed Therapies GPRC5D-directed Therapies Report Metrics
# Details
# Study Period
# 2020–2034

# """

# # Combine input text
# text = f"{title}\n{description}"

# # Define custom entity types you're interested in
# custom_labels = ["Disease", "Drug", "Organization", "Country"]

# # Predict entities using GLiNER
# entities = gliner_model.predict_entities(text, custom_labels, threshold=0.15, flat_ner=True)

# # Print extracted entities
# print("\n🔍 Extracted Entities:\n")
# for ent in entities:
#     print(f"{ent['text']} => {ent['label']} (start: {ent['start']}, end: {ent['end']})")

# print(f"\n✅ Total Entities Found: {len(entities)}")

################################################################################

# import re
# from gliner import GLiNER

# # ✅ Define your own chunking function
# def chunk_text(text, chunk_size=384, overlap=20):
#     """
#     Splits text into overlapping chunks using sentence boundaries.
#     """
#     sentences = re.split(r'(?<=[.!?]) +', text)
#     chunks = []
#     current_chunk = ""

#     for sentence in sentences:
#         if len(current_chunk) + len(sentence) <= chunk_size:
#             current_chunk += sentence + " "
#         else:
#             chunks.append(current_chunk.strip())
#             current_chunk = sentence + " "

#     if current_chunk:
#         chunks.append(current_chunk.strip())

#     # Add overlap between chunks
#     if overlap > 0 and len(chunks) > 1:
#         overlapped_chunks = []
#         for i in range(len(chunks)):
#             current = chunks[i]
#             if i > 0:
#                 prev = chunks[i - 1].split()[-overlap:]
#                 current = " ".join(prev) + " " + current
#             overlapped_chunks.append(current.strip())
#         return overlapped_chunks

#     return chunks

# # ✅ Load the GLiNER model
# gliner_model = GLiNER.from_pretrained("urchade/gliner_largev2")

# # ✅ Input text (title + description)
# title = "Berger Montague PC Investigates Rocket Pharmaceuticals (RCKT) Over Safety Disclosures and Stock Drop"
# description = """
# The GPRC5D-directed therapies market is witnessing robust growth, driven by the high unmet need in relapsed/refractory multiple myeloma. Innovative modalities like CAR-T cells, bispecific antibodies, and antibody-drug conjugates targeting GPRC5D are advancing rapidly through clinical pipelines. With limited cross-reactivity to normal tissues, GPRC5D presents a promising target, enhancing its commercial potential.
# LAS VEGAS , July 3, 2025 /PRNewswire/ -- DelveInsight's GPRC5D-directed Therapies Market Size, Target Population, Competitive Landscape & Market Forecast report includes a comprehensive understanding of current treatment practices, emerging GPRC5D-directed therapies, market share of individual therapies, and current and forecasted GPRC5D-directed therapies market size from 2020 to 2034, segmented into 7MM [ the United States , the EU4 ( Germany , France , Italy , and Spain ), the United Kingdom , and Japan ].
# Key Takeaways from the GPRC5D-directed Therapies Market Report
# As per DelveInsight's analysis, the total market size of GPRC5D-directed therapies in the 7MM is expected to surge significantly by 2034. As per DelveInsight's estimates, the total market size in the US for multiple myeloma was ~USD 15 billion in 2024, which is expected to show positive growth. Currently, TALVEY (talquetamab-tgvs) , approved in August 2023 , is the first and the only approved GPRC5D-directed therapy in the market. In the US, there were around 33,700 incident cases of multiple myeloma in 2024, which was the highest among seven major markets. Leading GPRC5D-directed therapies companies, such as Bristol Myers Squibb (Juno Therapeutics), AstraZeneca, Oricell, CARsgen Therapeutics, Sanofi, Jiangsu Simcere Pharmaceutical, AbbVie, and others, are developing novel GPRC5D-directed therapies that can be available in the GPRC5D-directed therapies market in the coming years. Some of the key GPRC5D-directed therapies in the pipeline include Arlocabtagene autoleucel (BMS-986393), AZD0305, OriCAR-017, CT071, SAR446523 , SIM0500, and others. In June 2025 , CARsgen Therapeutics presented a Phase I study of CT071 for the treatment of newly diagnosed multiple myeloma at the European Hematology Association (EHA) website. Discover which indication is expected to grab the major GPRC5D-directed therapies market share @ GPRC5D-directed Therapies Market Report
# GPRC5D-directed Therapies Market Dynamics
# The GPRC5D-directed therapies market is emerging as a significant focus within the landscape of multiple myeloma treatment. GPRC5D is an orphan receptor predominantly expressed on malignant plasma cells, with limited expression in normal tissues, making it an attractive therapeutic target. Its expression is independent of BCMA , allowing it to serve as a complementary or alternative target, particularly in patients who have relapsed after BCMA-directed treatments. This has driven a wave of innovation in bispecific antibodies, ADCs, and CAR-T cell therapies targeting GPRC5D .
# The competitive landscape is rapidly evolving, with multiple biotech and pharmaceutical companies advancing GPRC5D-targeted candidates through preclinical and clinical pipelines. Notably, companies such as Johnson & Johnson (via its teclistamab platform) and Legend Biotech are exploring GPRC5D-directed bispecifics and CAR-T constructs, respectively. Early-phase clinical trials have shown promising efficacy in relapsed/refractory multiple myeloma (RRMM) populations , even among those previously treated with BCMA-targeted agents. The ability of GPRC5D-targeting therapies to produce deep and durable responses , with manageable toxicity profiles, is a key factor fueling investor interest and partnerships in this space.
# Market dynamics are further shaped by the unmet clinical need in late-line multiple myeloma settings . As more patients receive BCMA therapies earlier in their treatment journeys, resistance and relapse rates have prompted the search for alternative targets like GPRC5D. Regulatory agencies , including the FDA and EMA, have shown openness to accelerated pathways for novel immunotherapies in RRMM, creating favorable conditions for GPRC5D-focused assets. Additionally, the ongoing development of allogeneic platforms and off-the-shelf bispecific antibodies is poised to enhance patient access and reduce logistical burdens compared to autologous CAR-T approaches.
# However, challenges remain in this emerging space. Differentiation among therapies targeting the same antigen will be crucial, as companies aim to optimize dosing schedules, minimize on-target off-tumor toxicities, and address manufacturing scalability. Pricing and reimbursement strategies will also play a critical role, particularly as competition intensifies and cost-effectiveness comes under scrutiny from payers and health systems.
# Looking forward, the GPRC5D-directed therapy market is expected to expand significantly over the next 10 years, driven by a combination of clinical validation, rising relapse rates in existing treatment paradigms, and an influx of new therapeutic modalities . Success in this domain will likely depend on the ability to integrate GPRC5D-targeted agents into combination regimens, leverage companion diagnostics to optimize patient selection, and establish long-term safety and efficacy through robust Phase III trials and real-world evidence.
# GPRC5D-directed Therapies Treatment Market
# GPRC5D is highly expressed in multiple myeloma cells and is found in abundance within the bone marrow of patients diagnosed with multiple myeloma or smoldering multiple myeloma. Currently, TALVEY (talquetamab-tgvs) , approved in August 2023 , is the first and the only approved GPRC5D-directed therapy in the market. TALVEY is a novel, first-in-class bispecific T-cell engager that simultaneously binds to the CD3 receptor on T cells and GPRC5D, which is present on multiple myeloma cells, non-cancerous plasma cells, and certain healthy epithelial tissues, particularly in the skin and tongue. It is administered subcutaneously on a weekly or biweekly basis following an initial step-up dosing phase, giving physicians the flexibility to tailor dosing schedules to individual patient needs.
# Learn more about the FDA-approved GPRC5D-directed therapies @ Approved GPRC5D-directed Therapies
# Key Emerging GPRC5D-directed Therapies and Companies
# Some of the major emerging key players in the GPRC5D market include CARsgen Therapeutics (CT071), Oricell (OriCAR-017), Bristol- Myers Squibb (Arlocabtagene autoleucel), AstraZeneca (AZD0305), and others.
# OriCAR-017 is an innovative CAR T cell therapy developed to target GPRC5D, aimed at treating relapsed or refractory multiple myeloma. Developed using Oricell Therapeutics' advanced proprietary technology, OriCAR-017 demonstrates unique advantages in terms of binding strength, cell persistence, tumor-fighting capability, and safety. It received Investigational New Drug (IND) approval from the U.S. FDA in January 2024 , following earlier approval by China's NMPA in 2023.
# BMS-986393 is an autologous CAR T cell therapy also directed at GPRC5D. Currently under evaluation in the Phase II QUINTESSENTIAL trial for relapsed/refractory multiple myeloma, the therapy is expected to yield key data in 2026. At the ASH 2023 conference, updated results from its Phase I trial highlighted durable responses, manageable safety, and clinical activity in patients regardless of prior BCMA-targeted treatment exposure.
# The anticipated launch of these emerging therapies are poised to transform the GPRC5D-directed therapies market landscape in the coming years. As these cutting-edge therapies continue to mature and gain regulatory approval, they are expected to reshape the GPRC5D-directed therapies market landscape, offering new standards of care and unlocking opportunities for medical innovation and economic growth.
# To know more about GPRC5D-directed therapies clinical trials, visit @ GPRC5D-directed Therapies Treatment
# GPRC5D-directed Therapies Overview
# GPRC5D, an orphan G-protein-coupled receptor located on chromosome 12p13, lacks a known ligand. It has emerged as a validated therapeutic target in multiple myeloma. Early-phase clinical trials involving T–cell–reducing therapies aimed at GPRC5D have demonstrated encouraging efficacy along with tolerable safety profiles, although these findings still require confirmation in Phase III studies.
# This receptor is highly expressed in multiple myeloma cells and is prevalent in the bone marrow of patients with both active and smoldering disease. Furthermore, its mRNA levels are notably higher in multiple myeloma compared to other hematologic malignancies. The selective overexpression of GPRC5D in myeloma cells makes it an attractive candidate for immune effector cell–based therapies. Targeting GPRC5D, either as a standalone treatment or in combination with other anti-myeloma agents, could broaden the therapeutic landscape, particularly for patients who require novel mechanisms of action, whether to delay the use of BCMA-targeted therapies, overcome poor responses or antigen loss, or address clonal variability.
# Although GPRC5D and BCMA are both expressed on CD138+ myeloma cells, their expression is not interdependent, making them distinct therapeutic targets. Importantly, GPRC5D expression remains stable even when BCMA is lost, a known resistance mechanism in some relapsed patients, supporting the rationale for combining GPRC5D- and BCMA-directed T-cell therapies to better manage the disease's heterogeneity.
# GPRC5D-directed Therapies Epidemiology Segmentation
# The GPRC5D-directed therapies market report proffers epidemiological analysis for the study period 2020–2034 in the 7MM, segmented into:
# Total Cases in Selected Indications (Multiple Myeloma) for GPRC5D-directed Therapies Total Eligible Patient Pool in Selected Indications for GPRC5D-directed Therapies Total Treated Cases in Selected Indications for GPRC5D-directed Therapies GPRC5D-directed Therapies Report Metrics
# Details
# Study Period
# 2020–2034
# GPRC5D-directed Therapies Report Coverage
# 7MM [The United States, the EU-4 (Germany, France, Italy, and Spain), the United Kingdom, and Japan]
# Key GPRC5D-directed Therapies Companies
# Bristol Myers Squibb (Juno Therapeutics), AstraZeneca, Oricell, CARsgen Therapeutics, Sanofi, Jiangsu Simcere Pharmaceutical, AbbVie, Johnson & Johnson Innovative Medicine (Janssen Biotech), and others
# Key GPRC5D-directed Therapies
# Arlocabtagene autoleucel (BMS-986393), AZD0305, OriCAR-017, CT071, SAR446523, SIM0500, TALVEY, and others
# Scope of the GPRC5D-directed Therapies Market Report
# GPRC5D-directed Therapies Therapeutic Assessment: GPRC5D-directed Therapies current marketed and emerging therapies GPRC5D-directed Therapies Market Dynamics: Conjoint Analysis of Emerging GPRC5D-directed Therapies Drugs Competitive Intelligence Analysis: SWOT analysis and Market entry strategies Unmet Needs, KOL's views, Analyst's views, GPRC5D-directed Therapies Market Access and Reimbursement Discover more about GPRC5D-directed therapies in development @ GPRC5D-directed Therapies Clinical Trials
# Table of Contents
# 1.
# Key Insights
# 2.
# Report Introduction
# 3.
# Executive Summary
# 4.
# Key Events
# 5.
# Market Forecast Methodology
# 6.
# GPRC5D-directed Therapies Market Overview at a Glance in the 7MM
# 6.1.
# Market Share (%) Distribution by Therapies in 2025
# 6.2.
# Market Share (%) Distribution by Therapies in 2034
# 6.3.
# Market Share (%) Distribution by Indications in 2025
# 6.4.
# Market Share (%) Distribution by Indications in 2034
# 7.
# GPRC5D-directed Therapies: Background and Overview
# 7.1.
# Introduction
# 7.2.
# The Potential of GPRC5D-directed Therapies in Different Indications
# 7.3.
# Clinical Applications of GPRC5D-directed Therapies
# 8.
# Target Patient Pool of GPRC5D-directed Therapies
# 8.1.
# Assumptions and Rationale
# 8.2.
# Key Findings
# 8.3.
# Total Cases of Selected Indication for GPRC5D-directed Therapies in the 7MM
# 8.4.
# Total Eligible Patient Pool of Selected Indication for GPRC5D-directed Therapies in the 7MM
# 8.5.
# Total Treatable Cases in Selected Indication for GPRC5D-directed Therapies in the 7MM
# 9.
# Marketed Therapies
# 9.1.
# Key Competitors
# 9.2.
# TALVEY (talquetamab-tgvs): Johnson & Johnson Innovative Medicine
# 9.2.1.
# Product Description
# 9.2.2.
# Regulatory milestones
# 9.2.3.
# Other developmental activities
# 9.2.4.
# Clinical development
# 9.2.5.
# Safety and efficacy
# List to be continued in the report
# 10.
# Emerging Therapies
# 10.1.
# Key Competitors
# 10.2.
# OriCAR-017: Oricell
# 10.2.1.
# Product Description
# 10.2.2.
# Other developmental activities
# 10.2.3.
# Clinical development
# 10.2.4.
# Safety and efficacy
# 10.3.
# Arlocabtagene autoleucel (BMS-986393): Bristol Myers Squibb (Juno Therapeutics)
# 10.3.1.
# Product Description
# 10.3.2.
# Other developmental activities
# 10.3.3.
# Clinical development
# 10.3.4.
# Safety and efficacy
# List to be continued in the report
# 11.
# GPRC5D-directed Therapies: Seven Major Market Analysis
# 11.1.
# Key Findings
# 11.2.
# Market Outlook
# 11.3.
# Conjoint Analysis
# 11.4.
# Key Market Forecast Assumptions
# 11.4.1.
# Cost Assumptions and Rebates
# 11.4.2.
# Pricing Trends
# 11.4.3.
# Analogue Assessment
# 11.4.4.
# Launch Year and Therapy Uptakes
# 11.5.
# Total Market Sizes of GPRC5D-directed Therapies by Indications in the 7MM
# 11.6.
# The United States Market Size
# 11.6.1.
# Total Market Size of GPRC5D-directed Therapies in the United States
# 11.6.2.
# Market Size of GPRC5D-directed Therapies by Indication in the United States
# 11.6.3.
# Market Size of GPRC5D-directed Therapies by Therapies in the United States
# 11.7.
# EU4 and the UK
# 11.7.1.
# Total Market Size of GPRC5D-directed Therapies in EU4 and the UK
# 11.7.2.
# Market Size of GPRC5D-directed Therapies by Indications in EU4 and the UK
# 11.7.3.
# Market Size of GPRC5D-directed Therapies by Therapies in EU4 and the UK
# 11.8.
# Japan
# 11.8.1.
# Total Market Size of GPRC5D-directed Therapies Inhibitors in Japan
# 11.8.2.
# Market Size of GPRC5D-directed Therapies by Indications in Japan
# 11.8.3.
# Market Size of GPRC5D-directed Therapies by Therapies in Japan
# 12.
# SWOT Analysis
# 13.
# KOL Views
# 14.
# Unmet Needs
# 15.
# Market Access and Reimbursement
# 16.
# Appendix
# 16.1.
# Bibliography
# 16.2.
# Report Methodology
# 17.
# DelveInsight Capabilities
# 18.
# Disclaimer
# 19.
# About DelveInsight
# Related Reports
# Multiple Myeloma Market
# Multiple Myeloma Market Insights, Epidemiology, and Market Forecast – 2034 report delivers an in-depth understanding of the disease, historical and forecasted epidemiology, as well as the market trends, market drivers, market barriers, and key multiple myeloma companies, including Sanofi, Karyopharm Therapeutics, AbbVie, Takeda Pharmaceutical, Celgene, Bristol-Myers Squibb, RAPA Therapeutics, Pfizer, Array Biopharma, Cellectar Biosciences, BioLineRx, Celgene, Aduro Biotech, ExCellThera, Janssen Pharmaceutical, Precision BioSciences, Takeda, Glenmark (Ichnos Sciences SA), Poseida Therapeutics, Molecular Partners AG, Chipscreen Biosciences, AbbVie, Genentech (Roche), Janssen Biotech, Nanjing Legend Biotech, Merck Sharp & Dohme Corp., among others.
# CAR-T Market
# CAR-T Market Size, Target Population, Competitive Landscape & Market Forecast – 2034 report delivers an in-depth understanding of the market trends, market drivers, market barriers, and key CAR-T companies, including Gilead Sciences, Inc., Novartis AG, Caribou Biosciences, Inc., Aurora Biopharma, Bristol-Myers Squibb Company, CARsgenTherapeutics Co., Ltd, JW Therapeutics ( Shanghai ) Co., Ltd., Cartesian Therapeutics, Inc., Johnson & Johnson Services, Inc. (Janssen Global Services, LLC), among others.
# Antibody Drug Conjugates Market
# Antibody Drug Conjugates Market Size, Target Population, Competitive Landscape & Market Forecast – 2034 report delivers an in-depth understanding of the market trends, market drivers, market barriers, and key ADC companies, including NBE-Therapeutics, ImmunoGen, Inc., Seagen Inc., ADC Therapeutics, Mythic Therapeutics, Sutro Biopharma, Merck KGaA, Sorrento Therapeutics, Inc., Peak Bio, Regeneron Pharmaceuticals, Asana BioSciences, Tanabe Research Laboratories USA , OBI Pharma, Sanofi, Navrogen, Inc., among others.
# Bispecifics/Trispecifics Market
# Bispecifics/Trispecifics Market Forecast and Competitive Landscape – 2035 report delivers an in-depth understanding of the market trends, market drivers, market barriers, and key bispecifics/trispecifics companies, including Janssen, Amgen, Akeso, Zymeworks, Roche, IGM Biosciences, MacroGenics, Provention Bio, Jiangsu Alphamab Biopharmaceuticals, Sichuan Baili Pharmaceutical, Regeneron Pharmaceuticals, Boehringer Ingelheim, among others.
# About DelveInsight
# DelveInsight is a leading Business Consultant and Market Research firm focused exclusively on life sciences. It supports pharma companies by providing comprehensive end-to-end solutions to improve their performance. Get hassle-free access to all the healthcare and pharma market research reports through our subscription-based platform PharmDelve .
# Contact Us
# Shruti Thakur [email protected] +14699457679
# Logo: https://mma.prnewswire.com/media/1082265/3528414/DelveInsight_Logo.jpg
# SOURCE DelveInsight Business Research, LLP
# WANT YOUR COMPANY'S NEWS FEATURED ON PRNEWSWIRE.COM? 440k+
# Newsrooms &
# Influencers 9k+
# Digital Media
# Outlets 270k+
# Journalists
# Opted In GET STARTED
# """

# # ✅ Combine title + description
# full_text = f"{title}\n{description}"

# # ✅ Labels to extract
# custom_labels = ["Disease", "Drug", "Organization", "Country"]

# # ✅ Chunk text safely
# chunks = chunk_text(full_text, chunk_size=384)

# # ✅ Extract entities from each chunk
# all_entities = []
# for chunk in chunks:
#     entities = gliner_model.predict_entities(
#         chunk,
#         custom_labels,
#         threshold=0.15,
#         flat_ner=True
#     )
#     all_entities.extend(entities)

# # ✅ Print extracted results
# print("\n🔍 Extracted Entities:\n")
# for ent in all_entities:
#     print(f"{ent['text']} => {ent['label']} (start: {ent['start']}, end: {ent['end']})")

# print(f"\n✅ Total Entities Found: {len(all_entities)}")
