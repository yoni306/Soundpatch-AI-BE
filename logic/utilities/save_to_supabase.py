import os
from config import settings

def save_to_supabase(file_path: str, supabase_client, bucket_name: str) -> str:
    """
    Save a file to Supabase storage.
    
    Args:
        file_path (str): Path to the file to save
        is_processed (bool): Whether this is a processed file (True) or original file (False)
    
    Returns:
        str: The file key in Supabase storage
    """    
    # Get the file name from the path
    file_name = os.path.basename(file_path)
        
    # Read the file
    with open(file_path, 'rb') as f:
        file_data = f.read()
    
    # Upload the file to Supabase storage
    result = supabase_client.storage.from_(bucket_name).upload(
        file_name,
        file_data
    )
    
    # Get the public URL
    file_key = f"{bucket_name}/{file_name}"
    
    return file_key 