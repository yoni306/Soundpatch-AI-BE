import os
import mimetypes
from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import Response
from config import settings
from logic.factory.model_factory import ModelFactory
from process_flow import process_file
from typing import Union

router = APIRouter()

# Ensure upload directory exists
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)

# Initialize models
(
    detect_noise_model,
    gemini_model,
    wav2vec2_processor,
    wav2vec2_model,
    text_to_mel_model,
    hifigan_model,
    device
) = ModelFactory.setup_models()

# Infer MIME type
def get_mime_type(filename: str) -> str:
    mime_type, _ = mimetypes.guess_type(filename)
    return mime_type or "application/octet-stream"

@router.post("/process")
async def process_file_upload(file: UploadFile = File(...)) -> Response:
    """
    Process an uploaded audio/video file and return the processed version.
    
    Args:
        file (UploadFile): The file to process
        
    Returns:
        Response: The processed file content with appropriate headers
    """
    try:
        # Step 1: Save uploaded file temporarily
        local_input_path = os.path.join(settings.UPLOAD_DIR, file.filename)
        with open(local_input_path, 'wb') as f:
            content = await file.read()
            f.write(content)

        # Step 2: Process file
        processed_content = process_file(
            local_input_path,
            detect_noise_model,
            gemini_model,
            wav2vec2_processor,
            wav2vec2_model,
            text_to_mel_model,
            hifigan_model,
            device
        )

        # Step 3: Get content type
        content_type = get_mime_type(file.filename)

        # Step 4: Clean up input file
        os.remove(local_input_path)

        # Step 5: Return the processed file content
        return Response(
            content=processed_content,
            media_type=content_type,
            headers={
                "Content-Disposition": f"attachment; filename=processed_{file.filename}"
            }
        )

    except Exception as e:
        print("❌ Error:", str(e))
        raise HTTPException(status_code=500, detail="Failed to process file.") 