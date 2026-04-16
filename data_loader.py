import pandas as pd
import os
import json

DATA_DIR = "data"

def load_dataframes():
    """Loads all CSVs in the data directory into a dictionary of DataFrames."""
    dfs = {}
    print("Loading datasets into memory...")
    for file in os.listdir(DATA_DIR):
        if file.endswith(".csv"):
            # Clean up the table name for easier querying
            table_name = file.replace("olist_", "").replace("_dataset.csv", "").replace(".csv", "")
            file_path = os.path.join(DATA_DIR, file)
            
            # Load the CSV
            dfs[table_name] = pd.read_csv(file_path)
            print(f"  -> Loaded '{table_name}' | Shape: {dfs[table_name].shape}")
            
    # Enrich tables with product category translations for easier querying
    if "products" in dfs and "product_category_name_translation" in dfs:
        dfs["products"] = dfs["products"].merge(
            dfs["product_category_name_translation"],
            on="product_category_name",
            how="left"
        )
        print("  -> Enriched 'products' with English category names")

    if "order_items" in dfs and "products" in dfs:
        dfs["order_items"] = dfs["order_items"].merge(
            dfs["products"][["product_id", "product_category_name", "product_category_name_english"]],
            on="product_id",
            how="left"
        )
        print("  -> Enriched 'order_items' with product category info")

    return dfs

def generate_schema(dfs):
    """Generates a schema dictionary mapping table names to columns and sample data."""
    print("\nExtracting schema and sample data...")
    schema = {}
    for name, df in dfs.items():
        # Handle datetime columns automatically if possible, otherwise string representations
        schema[name] = {
            "columns": df.dtypes.astype(str).to_dict(),
            # Convert the first 3 rows to a dictionary format for the LLM to read
            "sample_data": df.head(3).to_dict(orient="records") 
        }
    return schema

if __name__ == "__main__":
    # 1. Load the data
    my_dataframes = load_dataframes()
    
    # 2. Generate the schema
    db_schema = generate_schema(my_dataframes)
    
    # 3. Save the schema to a JSON file
    with open("schema.json", "w") as f:
        json.dump(db_schema, f, indent=4)
        
    print("\nSuccess! Schema saved to schema.json")