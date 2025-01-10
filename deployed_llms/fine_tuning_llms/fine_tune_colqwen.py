import os
import torch
from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from colpali_engine.models import ColQwen2, ColQwen2Processor
from datasets import Dataset
from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from tqdm import tqdm
from PyPDF2 import PdfReader
import openai

# Set up environment variables for API keys
os.environ["OPENAI_API_KEY"] = "your_openai_api_key"
openai.api_key = os.getenv("OPENAI_API_KEY")

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Function to generate synthetic data using Claude-3.5 Sonnet
def generate_synthetic_data(prompt):
    response = openai.Completion.create(
        engine="claude-3.5-sonnet",
        prompt=prompt,
        max_tokens=150
    )
    return response.choices[0].text.strip()

# Load and process PDF documents
pdf_dir = "path_to_pdf_directory"
pdf_files = [os.path.join(pdf_dir, f) for f in os.listdir(pdf_dir) if f.endswith('.pdf')]

# Extract text from PDFs
documents = []
for pdf_file in tqdm(pdf_files, desc="Extracting text from PDFs"):
    text = extract_text_from_pdf(pdf_file)
    documents.append(text)

# Vectorize documents
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(documents)

# Initialize Active Learner with Support Vector Classifier
learner = ActiveLearner(
    estimator=make_pipeline(TfidfVectorizer(), SVC(probability=True)),
    query_strategy=uncertainty_sampling,
    X_training=X,
    y_training=None  # Assuming unsupervised initial training
)

# Active learning loop to select informative samples
n_queries = 10
data_samples = []
for _ in range(n_queries):
    query_idx, query_instance = learner.query(X)
    learner.teach(X[query_idx], None)  # Simulated labeling process

    prompt = f"Generate a relevant query for the following content:\n\n{documents[query_idx[0]]}"
    synthetic_query = generate_synthetic_data(prompt)
    data_samples.append({"query": synthetic_query, "document": documents[query_idx[0]]})

# Prepare dataset for fine-tuning
dataset = Dataset.from_list(data_samples)

# Load ColQwen2 model and processor
model_name = "vidore/colqwen2-v0.1"
model = ColQwen2.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
).eval()

processor = ColQwen2Processor.from_pretrained(model_name)

# Apply 8-bit quantization
quantization_config = BitsAndBytesConfig(load_in_8bit=True)
model = model.quantize(quantization_config)

# Configure LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none"
)
model = get_peft_model(model, lora_config)

# Fine-tune the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.train()

train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
num_epochs = 3

for epoch in range(num_epochs):
    epoch_loss = 0.0
    for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}"):
        inputs = processor(batch["document"], return_tensors='pt', padding=True).to(device)
        labels = batch["query"]  # Replace with appropriate label preprocessing

        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        epoch_loss += loss.item()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss/len(train_dataloader)}")

# Save the fine-tuned model
output_dir = "path_to_save_fine_tuned_model"
model.save_pretrained(output_dir)
processor.save_pretrained(output_dir)

