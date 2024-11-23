from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
from typing import List
import base64
import json
import uvicorn
from vlm_model import PDFEmbeddingCreator

app = FastAPI()


# Updated ImageRequest class to represent the structure of the JSON file
class ImageRequest(BaseModel):
    images: List[str]  # This can still be used for validation if needed


@app.post("/process_images")
async def process_images(file: UploadFile = File(...)):
    try:
        # Read the JSON file
        contents = await file.read()
        image_data = json.loads(contents)

        # Extract images from the JSON data
        image_bytes_list = []
        for image_str in image_data.get("images", []):
            try:
                image_bytes = base64.b64decode(image_str)
                image_bytes_list.append(image_bytes)
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid base64 string provided: {str(e)}"
                )
        
        # Send to mock backend
        result = PDFEmbeddingCreator.convert_byte_strings_to_images(image_bytes_list)
        
        return result
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing images: {str(e)}"
        )


IP_address: str = '127.0.0.1'
port: int = 8000
if __name__ == "__main__":
    uvicorn.run(app, host=IP_address, port=port)  
