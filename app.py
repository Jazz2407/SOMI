import os, uuid, json, threading, time, glob
from typing import Optional, List
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Response, Cookie
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn, ollama
from werkzeug.utils import secure_filename

from config import Config
from agents.agent_decision import process_query

config = Config()
app = FastAPI(title="SOMI Medical Assistant", version="2.0")

# Directories
UPLOAD_FOLDER = "uploads/backend"
SPEECH_DIR = "uploads/speech"
for directory in [UPLOAD_FOLDER, SPEECH_DIR, "data"]:
    os.makedirs(directory, exist_ok=True)

app.mount("/data", StaticFiles(directory="data"), name="data")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
templates = Jinja2Templates(directory="templates")

class QueryRequest(BaseModel):
    query: str
    conversation_history: List = []

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# --- UPDATED: Hospital Specialists Endpoint ---
@app.get("/hospital/{hospital_id}/specialists")
async def get_specialists(hospital_id: str):
    json_path = os.path.join("data", "hospital_specialists.json")
    if not os.path.exists(json_path):
        return {"status": "success", "specialists": []} # Fallback
    with open(json_path, "r") as f:
        data = json.load(f)
    return {"status": "success", "specialists": data.get(hospital_id, [])}

# --- UPDATED: Image Upload with Vision Response ---
# Update this block in your app.py
@app.post("/upload")
async def upload_image(image: UploadFile = File(...), text: str = Form("")):
    file_content = await image.read()
    
    # SYSTEM PROMPT: Forces the model to actually analyze diagnostic features
    medical_prompt = (
        f"Context: You are a Medical Vision AI. Analyze this skin lesion. "
        f"Identify specific features like 'annular plaques' or 'comedones'. "
        f"Question: {text if text else 'Identify this skin condition'}"
    )

    try:
        # PROBLEM FIX: Ensure 'images' is a list of bytes
        vision_res = ollama.generate(
            model='moondream', 
            prompt=medical_prompt, 
            images=[file_content]
        )
        return {"response": vision_res['response'], "status": "success"}
    except Exception as e:
        return {"response": f"AI Error: {str(e)}", "status": "error"}
    
    file_content = await image.read()
    
    # Save file temporarily for existing agent processing
    filename = secure_filename(f"{uuid.uuid4()}_{image.filename}")
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    with open(file_path, "wb") as f:
        f.write(file_content)
    
    try:
        # 1. REAL AI RESPONSE: Call Vision Model (Moondream/Phi-3)
        # Ensure you have 'ollama run moondream' installed locally
        vision_res = ollama.generate(
            model='moondream', 
            prompt=f"Identify the medical condition or skin disease in this image. Question: {text}", 
            images=[file_content]
        )
        ai_analysis = vision_res['response']

        # 2. PROMPT AGENT: Existing decision logic
        response_data = process_query({"text": text, "image": file_path})
        agent_text = response_data['messages'][-1].content
        
        result = {
            "status": "success",
            "response": f"**AI Analysis:** {ai_analysis}\n\n---\n**Agent Diagnosis:** {agent_text}", 
            "agent": response_data["agent_name"]
        }
        
        # Keep your existing segmentation image logic
        if "SKIN_LESION_AGENT" in response_data["agent_name"]:
            segmentation_path = os.path.join(SKIN_LESION_OUTPUT, "segmentation_plot.png")
            if os.path.exists(segmentation_path):
                result["result_image"] = f"/uploads/skin_lesion_output/segmentation_plot.png"
        
        os.remove(file_path)
        return result
    except Exception as e:
        if os.path.exists(file_path): os.remove(file_path)
        raise HTTPException(status_code=500, detail=str(e))
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)