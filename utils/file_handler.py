import pandas as pd
import os
from pathlib import Path
from typing import Optional

# NEW: Import LangChain Document Loaders
from langchain_community.document_loaders import TextLoader, Docx2txtLoader
# You might need to install 'unstructured' and other dependencies for Docx2txtLoader
# pip install "unstructured[docx]"

def read_file(filepath: Path) -> Optional[pd.DataFrame]:
    """Reads data from various file types into a pandas DataFrame."""
    ext = filepath.suffix.lower()

    if ext in ['.xls', '.xlsx']:
        print(f"Reading Excel file: {filepath}")
        return pd.read_excel(filepath)
    elif ext == '.csv':
        print(f"Reading CSV file: {filepath}")
        return pd.read_csv(filepath)
    elif ext == '.json':
        print(f"Reading JSON file: {filepath}")
        return pd.read_json(filepath)
    elif ext == '.txt':
        print(f"Reading Text file: {filepath}")
        try:
            loader = TextLoader(str(filepath))
            documents = loader.load()
            # For TXT, treat entire content as one description
            content = documents[0].page_content if documents else ""
            # Create a DataFrame with a default field name and the content as description
            file_name_stem = filepath.stem.replace('.', '_') # Replace dots in filename for a clean field name
            return pd.DataFrame({
                'Field Name': [f"{file_name_stem}_content"],
                'Description': [content]
            })
        except Exception as e:
            print(f"Error reading TXT file {filepath}: {e}")
            return None
    elif ext == '.docx':
        print(f"Reading DOCX file: {filepath}")
        try:
            loader = Docx2txtLoader(str(filepath))
            documents = loader.load()
            # For DOCX, treat entire content as one description
            content = " ".join([doc.page_content for doc in documents]) if documents else ""
            # Create a DataFrame with a default field name and the content as description
            file_name_stem = filepath.stem.replace('.', '_')
            return pd.DataFrame({
                'Field Name': [f"{file_name_stem}_content"],
                'Description': [content]
            })
        except Exception as e:
            print(f"Error reading DOCX file {filepath}: {e}")
            return None
    else:
        print(f"Unsupported file type: {ext}")
        return None

def write_excel_file(df: pd.DataFrame, output_filepath: Path):
    """Writes a pandas DataFrame to an Excel file."""
    df.to_excel(output_filepath, index=False)
    print(f"Data successfully written to {output_filepath}")