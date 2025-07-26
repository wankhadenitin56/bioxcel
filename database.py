import psycopg2
import pandas as pd

# Database connection details
host = "localhost"       
port = "5432"             
database = "PostgreSQL"
user = "prashant_dml"
password = "fghjDRFGH456?44?&jkk"

# Output file path
output_file = output_file = "C:/Users/EZB-VM-06/Desktop/webapp_1/NERWebApp/RunNER/helper_functions/Nitin_files/schema_table_row_counts.csv"
  

# Connect to PostgreSQL
try:
    conn = psycopg2.connect(
        host=host,
        port=port,
        database=database,
        user=user,
        password=password
    )
    print("Connection successful!")

    cursor = conn.cursor()

    # Get all schemas excluding system schemas
    cursor.execute("""
        SELECT schema_name
        FROM information_schema.schemata
        WHERE schema_name NOT IN ('pg_catalog', 'information_schema')
    """)
    schemas = [row[0] for row in cursor.fetchall()]

    result_data = []

    for schema in schemas:
        # Get all tables in the schema
        cursor.execute(f"""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = %s AND table_type = 'BASE TABLE'
        """, (schema,))
        tables = [row[0] for row in cursor.fetchall()]

        for table in tables:
            try:
                cursor.execute(f'SELECT COUNT(*) FROM "{schema}"."{table}"')
                row_count = cursor.fetchone()[0]
            except Exception as e:
                print(f"Error counting rows for {schema}.{table}: {e}")
                row_count = "Error"

            result_data.append({
                "schema_name": schema,
                "table_name": table,
                "row_count": row_count
            })

    # Convert to DataFrame and save to CSV
    df = pd.DataFrame(result_data)
    print(df)
    df.to_csv(output_file, index=False)
    print(f"\nRow counts saved to {output_file}")

except Exception as e:
    print(f"Error connecting to database: {e}")

finally:
    if 'conn' in locals() and conn:
        conn.close()
