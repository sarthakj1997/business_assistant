import tempfile
from fastapi import APIRouter, UploadFile, File
from backend.services.parse_invoice import extract_with_layout
from backend.services.process_invoice import process_invoice, InvoiceSummary
from backend.services.upload_to_db_invoice import save_invoice_to_db

# Then use it:

from loguru import logger

router = APIRouter()

@router.post("/pdf")
async def upload_pdf(file: UploadFile = File(...), user_id: int = 1):

    logger.info(f"Received PDF: {file.filename}")
    
    # Save file to a temporary location
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    file_content = await file.read()
    temp_file.write(file_content)
    temp_file.close()
    temp_path = temp_file.name

    # Process the saved PDF file
    pages = extract_with_layout(temp_path)
    structured_data = process_invoice(pages)

    #save to database
    invoice_id = save_invoice_to_db(structured_data, user_id)


    return {
        "invoice_id": invoice_id,
        "structured_data": structured_data
    }