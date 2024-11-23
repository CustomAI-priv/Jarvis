# Prediction interface for Cog ⚙️
# https://cog.run/python

from cog import BasePredictor, Path, Input
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

 
class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        
        # get the model from 
        self.model_name: str = "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF"

        # download the model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
 
    def predict(self,
            prompt: str = Input(description="Prompt for the LLM"),
    ) -> str:
        """Run a single prediction on the model"""

        # format the prompt
        messages: list = [{"role": "user", "content": prompt}]

        # tokenize the message
        tokenized_message = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True)
        
        # generate the response
        response_token_ids = self.model.generate(tokenized_message['input_ids'].cuda(),attention_mask=tokenized_message['attention_mask'].cuda(),  max_new_tokens=4096, pad_token_id = self.tokenizer.eos_token_id)
        
        # decode the response
        generated_tokens = response_token_ids[:, len(tokenized_message['input_ids'][0]):]
        generated_text: str = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        
        return generated_text
                