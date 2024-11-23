from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from typing import AsyncGenerator
import uvicorn

app = FastAPI()

# define a paragraph of text to stream
text = "This is a test paragraph to stream. It contains some text that will be streamed to the client. The client will receive this text in chunks, and the chunks will be displayed as they are received. This is a test paragraph to stream. It contains some text that will be streamed to the client. The client will receive this text in chunks, and the chunks will be displayed as they are received. This is a test paragraph to stream. It contains some text that will be streamed to the client. The client will receive this text in chunks, and the chunks will be displayed as they are received."

async def text_generator(text: str) -> AsyncGenerator[str, None]:
    """Generate text chunks for streaming"""
    # Split text into smaller chunks (e.g., by sentences or fixed length)
    chunk_size = 100  # Adjust chunk size as needed
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]
        yield chunk
        # Add small delay between chunks if desired
        # await asyncio.sleep(0.1)

@app.get("/stream")
async def stream_text():
    """Stream text content to client"""
    return StreamingResponse(
        text_generator(text),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
