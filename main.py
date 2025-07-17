from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import os
import shutil
import pandas as pd
import time

from utils.file_handler import read_file, write_excel_file
# Removed specific_field_standardizer import as it's now internal to matcher.py
# from utils.text_standardizer import specific_field_standardizer
from utils.matcher import match_and_score_descriptions

app = FastAPI()

# Define directories for uploads and downloads
UPLOAD_DIR = Path("uploads")
DOWNLOAD_DIR = Path("downloads")

# Create directories if they don't exist
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Mount static files for HTML and potentially other assets
app.mount("/static", StaticFiles(directory="templates"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serves the main HTML page for file uploads."""
    with open(Path("templates") / "index.html", "r") as f:
        return HTMLResponse(content=f.read())

@app.post("/process_files/")
async def process_files(
    background_tasks: BackgroundTasks,
    file1: UploadFile = File(..., description="First input file (e.g., raw data)"),
    field_name_col1: str = Form("Field Name", description="Column name for field names in File 1"),
    description_col1: str = Form("Description", description="Column name for descriptions in File 1"),
    file2: UploadFile = File(..., description="Second input file (e.g., standardized lookup)"),
    standardized_name_col2: str = Form("Standardized Name", description="Column name for standardized names in File 2"),
    description_col2: str = Form("Description", description="Column name for descriptions in File 2"),
    min_score_threshold: float = Form(0.7, description="Minimum cosine similarity score for a match")
):
    """
    Processes two uploaded files:
    1. Standardizes field names in the first file using the specific "institution.x.x" format.
    2. Matches standardized names from file 1 with standardized names from file 2
       based on description similarity, calculates a match score, and generates an output file.
    """
    # --- Save File 1 ---
    file1_path = UPLOAD_DIR / file1.filename
    try:
        with open(file1_path, "wb") as buffer:
            shutil.copyfileobj(file1.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not save file1: {e}")
    finally:
        file1.file.close()

    # --- Save File 2 ---
    file2_path = UPLOAD_DIR / file2.filename
    try:
        with open(file2_path, "wb") as buffer:
            shutil.copyfileobj(file2.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not save file2: {e}")
    finally:
        file2.file.close()

    processing_time_seconds = 0

    try:
        # --- Process File 1: Read ---
        df1 = read_file(file1_path)
        if df1 is None:
            raise HTTPException(status_code=400, detail=f"Unsupported file type for {file1.filename}")

        # For .txt/.docx, file_handler creates a 'Field Name' column and 'Description'
        if file1_path.suffix.lower() in ['.txt', '.docx']:
            # The 'field_name_col1' and 'description_col1' from the form are ignored for these types
            # We use the default column names created by file_handler
            pass
        else:
            # For tabular files, validate user-provided column names
            if field_name_col1 not in df1.columns:
                raise HTTPException(status_code=400, detail=f"Column '{field_name_col1}' not found in {file1.filename}")
            if description_col1 not in df1.columns:
                df1[description_col1] = '' # Add empty description column if missing

        # --- Process File 2: Read Lookup Data ---
        df2 = read_file(file2_path)
        if df2 is None:
            raise HTTPException(status_code=400, detail=f"Unsupported file type for {file2.filename}")

        if standardized_name_col2 not in df2.columns:
            raise HTTPException(status_code=400, detail=f"Column '{standardized_name_col2}' not found in {file2.filename}")
        if description_col2 not in df2.columns:
            df2[description_col2] = '' # Create empty if missing


        # --- Time the Match and Score Descriptions phase ---
        start_time = time.monotonic()
        matched_df = match_and_score_descriptions(
            df1=df1,
            field_name_col1=field_name_col1, # Pass original field name column from form
            description_col1=description_col1, # Pass original description column from form
            df2=df2,
            standardized_name_col2=standardized_name_col2,
            description_col2=description_col2,
            min_score_threshold=min_score_threshold # Pass the slider value
        )
        end_time = time.monotonic()
        processing_time_seconds = round(end_time - start_time, 2)

        # --- Write Output File ---
        output_filename = f"matched_standardized_{Path(file1.filename).stem}.xlsx"
        output_filepath = DOWNLOAD_DIR / output_filename
        write_excel_file(matched_df, output_filepath)

        # Clean up uploaded files in the background
        background_tasks.add_task(os.remove, file1_path)
        background_tasks.add_task(os.remove, file2_path)

        return {
            "message": "Files processed successfully!",
            "download_filename": output_filename,
            "processing_time_seconds": processing_time_seconds
        }

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {e}")

@app.get("/download/{filename}")
async def download_file(filename: str):
    """Allows downloading of processed files."""
    filepath = DOWNLOAD_DIR / filename
    if not filepath.is_file():
        raise HTTPException(status_code=404, detail="File not found.")
    return FileResponse(path=filepath, filename=filename, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")