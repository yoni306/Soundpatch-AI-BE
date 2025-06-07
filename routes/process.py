import os
import shutil
import mimetypes
from fastapi import APIRouter, HTTPException, Form
from fastapi.responses import JSONResponse
from supabase import create_client, Client
from config import settings

router = APIRouter()

# Ensure upload directory exists
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)

# Initialize Supabase client
supabase: Client = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)

# Dummy processing function
def process_media_file(file_path: str) -> str:
    filename = os.path.basename(file_path)
    processed_path = os.path.join(settings.UPLOAD_DIR, f"processed_{filename}")
    shutil.copyfile(file_path, processed_path)  # Replace with real GPU logic
    return processed_path

    supabase_client = ClientFactory.setup_clients()

    # Get models from ModelFactory
    (
        detect_noise_model,
        gemini_model,
        wav2vec2_processor,
        wav2vec2_model,
        text_to_mel_model,
        hifigan_model,
        device
    ) = ModelFactory.setup_models()

    # Process the file
    process_file(
        filename,
        supabase_client,
        detect_noise_model,
        gemini_model,
        wav2vec2_processor,
        wav2vec2_model,
        text_to_mel_model,
        hifigan_model,
        device
    )


# Infer MIME type
def get_mime_type(filename: str) -> str:
    mime_type, _ = mimetypes.guess_type(filename)
    return mime_type or "application/octet-stream"

@router.post("/process")
async def process_file_from_supabase(file_key: str = Form(...)):
    try:
        # Step 1: Download file from Supabase
        local_input_path = os.path.join(settings.UPLOAD_DIR, os.path.basename(file_key))
        response = supabase.storage.from_(settings.STORAGE_BUCKET).download(file_key)

        with open(local_input_path, 'wb') as f:
            f.write(response)

        # Step 2: Process file
        processed_path = process_media_file(local_input_path)
        processed_filename = os.path.basename(processed_path)
        processed_key = f"processed/{processed_filename}"

        # Step 3: Detect MIME type
        content_type = get_mime_type(processed_filename)

        # Step 4: Upload processed file to Supabase
        with open(processed_path, 'rb') as f:
            supabase.storage.from_(settings.STORAGE_BUCKET).upload(
                processed_key,
                f,
                {"content-type": content_type}
            )

        # Step 5: Clean up
        os.remove(local_input_path)
        os.remove(processed_path)

        return JSONResponse(content={
            "message": "✅ File processed and uploaded.",
            "processed_file_key": processed_key,
            "mime_type": content_type
        })

    except Exception as e:
        print("❌ Error:", str(e))
        raise HTTPException(status_code=500, detail="Failed to process file.") 