# Prediction interface for Cog ⚙️
# https://cog.run/python

from cog import BasePredictor, Path, Input
import torch
from torch.utils.data import DataLoader as TorchDataLoader
from colpali_engine.models.paligemma import ColPali, ColPaliProcessor
from typing import cast
import base64
import io
from PIL import Image
from ast import literal_eval

 
class Predictor(BasePredictor):
    def setup(self) -> None:
        """Set up the ColPali model and processor"""

        # define the device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize the processor
        self.processor_colpali = cast(ColPaliProcessor, ColPaliProcessor.from_pretrained('vidore/colpali-v1.2'))

        # Initialize the model
        self.model = cast(
            ColPali,
            ColPali.from_pretrained(
                "vidore/colpali-v1.2",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map=self.device,  # Adjust as needed
            )
        )

        # Move the model to the device
        self.model.to(self.device)

    def _convert_base64_to_pil(self, base64_images: list[str]) -> list:
        """Convert base64 encoded images to PIL Image objects
        
        Args:
            base64_images (list[str]): List of base64 encoded image strings
            
        Returns:
            list: List of PIL Image objects
        """
        
        pil_images = []
        for b64_str in base64_images:
            # Remove data URL prefix if present
            if ',' in b64_str:
                b64_str = b64_str.split(',')[1]
                
            # Decode base64 string to bytes
            img_bytes = base64.b64decode(b64_str)
            
            # Convert bytes to PIL Image
            img = Image.open(io.BytesIO(img_bytes))
            pil_images.append(img)
            
        return pil_images

    def _convert_pil_to_base64(self, pil_image: Image.Image, add_url_prefix: bool = True) -> str:
        """Convert a PIL Image to base64 encoded string
        
        Args:
            pil_image (PIL.Image.Image): PIL Image object to convert
            add_url_prefix (bool): Whether to add data URL prefix. Defaults to True.
            
        Returns:
            str: Base64 encoded string representation of the image
        """
        # Create a bytes buffer for the image
        buffered = io.BytesIO()
        
        # Save image to the buffer in PNG format
        pil_image.save(buffered, format="PNG")
        
        # Get the base64 encoded string
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        # Add data URL prefix if requested
        if add_url_prefix:
            return f'data:image/png;base64,{img_str}'
        return img_str

    def predict(self, images: list[str], batch_size: int = 10) -> list[list[float]]:
        """Run a single prediction on the model"""

        # overwrite images with literal eval
        images = literal_eval(images)

        # convert base64 images to PIL images
        pil_images = self._convert_base64_to_pil(images)
        
        # Initialize list to store embeddings
        page_embeddings = []

        # Create a DataLoader over images
        dataloader = TorchDataLoader(
            pil_images,
            batch_size=batch_size,  # Using a reasonable default batch size
            shuffle=False,
            collate_fn=lambda batch: self.processor_colpali.process_images(batch),
        )

        # Process the pages and iterate over the dataloader object
        for batch_doc in dataloader:
            # Move batch to device
            batch_doc = {k: v.to(self.device) for k, v in batch_doc.items()}

            with torch.no_grad():
                # Forward pass
                outputs = self.model(**batch_doc)

            # Collect embeddings
            page_embeddings.extend(list(torch.unbind(outputs.to("cpu"))))

        # Convert tensors to lists before returning
        return [tensor.tolist() for tensor in page_embeddings]
