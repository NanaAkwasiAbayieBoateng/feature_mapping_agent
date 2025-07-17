import pandas as pd
from ollama import Client
import ollama
import numpy as np
from typing import List
from sklearn.metrics.pairwise import cosine_similarity # Import cosine_similarity from sklearn

# Initialize Ollama client
#ollama_client = Client()
OLLAMA_EMBEDDING_MODEL = 'mxbai-embed-large' 

# Helper function to get embeddings using ollama client
def get_embeddings(texts: List[str]) -> np.ndarray:
    """
    Generates embeddings for a list of texts using Ollama.
    Handles empty/non-string inputs by returning zero vectors.
    """
    embeddings = []
    for text in texts:
        if not isinstance(text, str) or not text.strip():
            embeddings.append([0.0] * 768) # Assuming 768 dimensions for mxbai-embed-large
            continue
        try:
            #response = ollama_client.embed(model=OLLAMA_EMBEDDING_MODEL, prompt=text)
            response =  ollama.embeddings(model=OLLAMA_EMBEDDING_MODEL, prompt=text)
            embeddings.append(response['embedding'])
        except Exception as e:
            print(f"Error getting embedding from Ollama for text '{text}': {e}")
            embeddings.append([0.0] * 768) # Return zero vector on error
    return np.array(embeddings)


def match_and_score_descriptions(
    df1: pd.DataFrame,
    field_name_col1: str, # This is the original 'Field Name' column from File 1
    description_col1: str,
    df2: pd.DataFrame,
    standardized_name_col2: str,
    description_col2: str,
    min_score_threshold: float = 0.6 # Default threshold, will be overridden by slider
) -> pd.DataFrame:
    """
    Matches field names from df1 with standardized names from df2 based on description cosine similarity.
    This function precisely replicates the logic and output columns of the user's provided snippet.
    """
    # Create copies to avoid SettingWithCopyWarning
    cdd_df = df1.copy()
    xob_df = df2.copy()

    # === Step 2: Clean and select relevant columns ===
    # Use dropna() as specified in the user's snippet.
    # This means rows with NaN in these specific columns will be removed.
    cdd_df = cdd_df[[field_name_col1, description_col1]].dropna()
    xob_df = xob_df[[standardized_name_col2, description_col2]].dropna()

    # Rename columns for internal consistency with the snippet's variable names
    cdd_df.rename(columns={field_name_col1: 'Field Name', description_col1: 'Description'}, inplace=True)
    xob_df.rename(columns={standardized_name_col2: 'StandardizedName', description_col2: 'Description'}, inplace=True)

    # Ensure descriptions are strings before passing to embedding model
    cdd_df['Description'] = cdd_df['Description'].astype(str)
    xob_df['Description'] = xob_df['Description'].astype(str)

    # === Step 3: Standardize Field Name (as per user's snippet logic) ===
    # Apply the exact lambda from the user's snippet.
    cdd_df['Standardized Field Name'] = cdd_df['Field Name'].apply(
        lambda x: f"institution.{str(x).strip().lower().replace(' ', '_')}.{str(x).strip().lower().replace(' ', '_')}"
    )

    # === Step 4: Generate embeddings using Ollama ===
    print("🔄 Generating embeddings for CDD descriptions...")
    cdd_embeddings = get_embeddings(cdd_df['Description'].tolist())

    print("🔄 Generating embeddings for XOB descriptions...")
    xob_embeddings = get_embeddings(xob_df['Description'].tolist())

    # Handle cases where one of the embedding sets might be empty after processing
    if cdd_embeddings.size == 0 or xob_embeddings.size == 0:
        print("One or both embedding sets are empty after dropna(). Cannot perform matching.")
        return pd.DataFrame(columns=[
            'Standardized Field Name',
            'StandardizedName',
            'Cosine Similarity',
            'Match',
            'Description' # This is Description from File 1
        ])

    # === Step 5: Compute cosine similarity matrix ===
    print("Computing cosine similarity matrix using sklearn.metrics.pairwise.cosine_similarity...")
    similarity_matrix = cosine_similarity(cdd_embeddings, xob_embeddings)

    # === Step 6: Find best matches ===
    best_indices = similarity_matrix.argmax(axis=1)
    best_scores = similarity_matrix.max(axis=1)

    # === Step 7: Assemble results ===
    results_data = []
    # Iterate through the indices of cdd_df to correctly align results
    for i in range(len(cdd_df)):
        # Get the best match index for the current cdd_df row
        best_match_idx_for_row = best_indices[i]
        
        # Ensure best_match_idx_for_row is valid for xob_df
        if len(xob_df) == 0 or best_match_idx_for_row >= len(xob_df) or best_match_idx_for_row < 0:
            # Fallback if no valid match or index is out of bounds
            matched_std_name_file2 = "N/A"
            score = 0.0
            match_status = 'no-match'
        else:
            # Correctly retrieve StandardizedName from xob_df using the specific index
            matched_std_name_file2 = xob_df.iloc[best_match_idx_for_row]['StandardizedName']
            score = best_scores[i]
            match_status = 'match' if score > min_score_threshold else 'no-match'

        results_data.append({
            'Standardized Field Name': cdd_df.iloc[i]['Standardized Field Name'],
            'StandardizedName': matched_std_name_file2,
            'Cosine Similarity': round(score, 4),
            'Match': match_status,
            'Description': cdd_df.iloc[i]['Description'] # This is Description from File 1
        })

    results = pd.DataFrame(results_data)

    # === Step 8: Sort results ===
    results = results.sort_values(by='Cosine Similarity', ascending=False)
    
    print("Matching complete.")
    return results


  