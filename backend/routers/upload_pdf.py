import tempfile
from fastapi import APIRouter, UploadFile, File
from backend.services.parse_invoice import extract_with_layout
from loguru import logger

router = APIRouter()

@router.post("/pdf")
async def upload_pdf(file: UploadFile = File(...)):
    logger.info(f"Received PDF: {file.filename}")
    
    # Save file to a temporary location
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    file_content = await file.read()
    temp_file.write(file_content)
    temp_file.close()
    temp_path = temp_file.name

    # Process the saved PDF file
    result = extract_with_layout(temp_path)
    return result