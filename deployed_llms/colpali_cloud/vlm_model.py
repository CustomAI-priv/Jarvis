from tqdm import tqdm
from torch.utils.data import DataLoader as TorchDataLoader
import torch
from pydantic import BaseModel
from io import BytesIO
from PIL import Image
import json


class ColPaliSettings(BaseModel):
    """Settings for the ColPali model"""

    device_map: str = 'auto'
    torch_type: str = 'torch.bfloat16'
    model_name: str = "vidore/colpali-v1.2"
    data_loader_batch_size: int = 2
    runtime_batch_size: int = 2
    target_hits_per_query_tensor: int = 20


class PDFEmbeddingCreator:
    """Class to create embeddings for PDF files."""

    def __init__(self):
        self.colpali_settings = ColPaliSettings()

    def convert_byte_strings_to_images(byte_strings: list) -> list:
        """Convert a list of byte strings representing images into PIL.Image objects."""
        images = []
        for byte_string in byte_strings:
            try:
                image = Image.open(BytesIO(byte_string))
                images.append(image)
            except Exception as e:
                print(f"Error converting byte string to image: {str(e)}")
        return images

    def create_embeddings(self, byte_strings) -> list:
        """Create the embeddings for the PDFs."""
        # convert the byte strings to images
        images = self.convert_byte_strings_to_images(byte_strings)
        
        # create the container for the embeddings
        page_embeddings = []

        # Create a DataLoader over images
        dataloader = TorchDataLoader(
            images,
            batch_size=self.colpali_settings.data_loader_batch_size,
            shuffle=False,
            collate_fn=lambda batch: self.processor_colpali.process_images(batch),
        )

        # Process the pages and iterate over the dataloader object
        for batch_doc in tqdm(dataloader, desc="Processing pages", leave=False):
            # Move batch to device
            batch_doc = {k: v.to(self.device) for k, v in batch_doc.items()}

            with torch.no_grad():
                # Forward pass
                outputs = self.model(**batch_doc)

            # Collect embeddings
            page_embeddings.extend(list(torch.unbind(outputs.to("cpu"))))

        return page_embeddings
    
    def embedding_facade(self, byte_strings) -> None:
        """Facade method to create embeddings and save them to a JSON file."""

        # create the embeddings
        embeddings = self.create_embeddings(byte_strings)
        
        # Convert embeddings to a list
        embeddings_list = [embedding.tolist() for embedding in embeddings]

        # Save to a JSON file
        with open('embeddings.json', 'w') as json_file:
            json.dump(embeddings_list, json_file)
