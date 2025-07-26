import chardet
import pandas as pd

# Step 1: Detect encoding
with open(r"C:/Users/EZB-VM-06/Downloads/daily_news_report_2025-07-10_270_records.csv" , 'rb') as f:
    result = chardet.detect(f.read(10000))  # Reads first 10KB to guess encoding

print("Detected Encoding:", result)

# Step 2: Read CSV with detected encoding
df = pd.read_csv(r'C:/Users/EZB-VM-06/Downloads/daily_news_report_2025-07-10_270_records.csv', encoding=result['encoding'])
