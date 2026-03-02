# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("fill-mask", model="medicalai/ClinicalBERT")


# Load model directly
from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("medicalai/ClinicalBERT")
model = AutoModelForMaskedLM.from_pretrained("medicalai/ClinicalBERT")