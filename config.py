"""
Configuration file for the Multi-Agent Medical Chatbot
ADAPTED FOR LOCAL OLLAMA & ON-DISK QDRANT (Optimized for 8GB RAM)
"""

import os
from dotenv import load_dotenv, find_dotenv
from langchain_ollama import ChatOllama, OllamaEmbeddings

# Load environment variables
load_dotenv(find_dotenv(), override=True)

# Shared local base URL for Ollama service (Running on your D: drive)
OLLAMA_BASE_URL = "http://localhost:11434"

class AgentDecisoinConfig:
    def __init__(self):
        # Using phi3 for lightweight local reasoning
        self.llm = ChatOllama(
            model="phi3",
            base_url=OLLAMA_BASE_URL,
            temperature=0.1
        )

class ConversationConfig:
    def __init__(self):
        # Higher temperature for natural chat
        self.llm = ChatOllama(
            model="phi3",
            base_url=OLLAMA_BASE_URL,
            temperature=0.7
        )

class WebSearchConfig:
    def __init__(self):
        self.llm = ChatOllama(
            model="phi3",
            base_url=OLLAMA_BASE_URL,
            temperature=0.3
        )
        self.context_limit = 20

class RAGConfig:
    def __init__(self):
        # --- STORAGE SETTINGS (Bypasses Docker) ---
        self.vector_db_type = "qdrant"
        self.use_local = True  # CRITICAL: Directly opens files from the D: drive
        self.vector_local_path = "./data/qdrant_db" 
        self.doc_local_path = "./data/docs_db"
        self.parsed_content_dir = "./data/parsed_docs"
        
        # --- VECTOR SETTINGS (Nomic Optimized) ---
        self.embedding_dim = 768 
        self.distance_metric = "Cosine"
        self.collection_name = "medical_assistance_rag"
        self.chunk_size = 512
        self.chunk_overlap = 50

        # --- OLLAMA LOCAL MODELS ---
        self.embedding_model = OllamaEmbeddings(
            model="nomic-embed-text",
            base_url=OLLAMA_BASE_URL
        )
        
        self.llm = ChatOllama(
            model="phi3",
            base_url=OLLAMA_BASE_URL,
            temperature=0.3
        )
        
        self.summarizer_model = ChatOllama(
            model="phi3",
            base_url=OLLAMA_BASE_URL,
            temperature=0.5
        )
        
        self.chunker_model = ChatOllama(
            model="phi3",
            base_url=OLLAMA_BASE_URL,
            temperature=0.0
        )
        
        self.response_generator_model = ChatOllama(
            model="phi3",
            base_url=OLLAMA_BASE_URL,
            temperature=0.3
        )
        
        # --- RETRIEVAL & RERANKING ---
        self.top_k = 5
        self.vector_search_type = 'similarity'
        self.huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
        self.reranker_model = "cross-encoder/ms-marco-TinyBERT-L-6"
        self.reranker_top_k = 3
        self.max_context_length = 8192
        self.include_sources = True
        self.min_retrieval_confidence = 0.40
        self.context_limit = 20
        
        # --- CLOUD FALLBACKS (Will be ignored by Local Mode) ---
        self.url = None
        self.api_key = None

class MedicalCVConfig:
    def __init__(self):
        self.brain_tumor_model_path = "./agents/image_analysis_agent/brain_tumor_agent/models/brain_tumor_segmentation.pth"
        self.chest_xray_model_path = "./agents/image_analysis_agent/chest_xray_agent/models/covid_chest_xray_model.pth"
        self.skin_lesion_model_path = "./agents/image_analysis_agent/skin_lesion_agent/models/checkpointN25_.pth.tar"
        self.skin_lesion_segmentation_output_path = "./uploads/skin_lesion_output/segmentation_plot.png"
        
        # Moondream (1.7GB) handles local multimodal vision
        self.llm = ChatOllama(
            model="moondream",
            base_url=OLLAMA_BASE_URL,
            temperature=0.1
        )

class SpeechConfig:
    def __init__(self):
        self.eleven_labs_api_key = os.getenv("ELEVEN_LABS_API_KEY")
        self.eleven_labs_voice_id = "21m00Tcm4TlvDq8ikWAM"

class ValidationConfig:
    def __init__(self):
        self.require_validation = {
            "CONVERSATION_AGENT": False,
            "RAG_AGENT": False,
            "WEB_SEARCH_AGENT": False,
            "BRAIN_TUMOR_AGENT": True,
            "CHEST_XRAY_AGENT": True,
            "SKIN_LESION_AGENT": True
        }
        self.validation_timeout = 300
        self.default_action = "reject"

class APIConfig:
    def __init__(self):
        self.host = "0.0.0.0"
        self.port = 8000
        self.debug = True
        self.rate_limit = 10
        self.max_image_upload_size = 5

class UIConfig:
    def __init__(self):
        self.theme = "light"
        self.enable_speech = True
        self.enable_image_upload = True

class Config:
    def __init__(self):
        self.agent_decision = AgentDecisoinConfig()
        self.conversation = ConversationConfig()
        self.rag = RAGConfig()
        self.medical_cv = MedicalCVConfig()
        self.web_search = WebSearchConfig()
        self.api = APIConfig()
        self.speech = SpeechConfig()
        self.validation = ValidationConfig()
        self.ui = UIConfig()
        
        self.eleven_labs_api_key = os.getenv("ELEVEN_LABS_API_KEY")
        self.tavily_api_key = os.getenv("TAVILY_API_KEY")
        self.max_conversation_history = 20