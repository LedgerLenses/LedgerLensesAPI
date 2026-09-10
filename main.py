import os
import json
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types

app = FastAPI(title="LedgerLenses API")

# CORS setup for Cloudflare
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    return {"status": "hot", "system": "LedgerLenses Backend Running"}

@app.post("/api/extract")
async def extract_document(file: UploadFile = File(...)):
    
    if file.content_type not in ["application/pdf", "image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Unsupported file format. Please upload PDF, JPG, or PNG.")

    # 1. Parse the rotating API key pool
    raw_keys = os.environ.get("GEMINI_API_KEY", "")
    api_keys = [k.strip() for k in raw_keys.split(",") if k.strip()]
    
    if not api_keys:
        raise HTTPException(status_code=500, detail="Server Configuration Error: No API keys found.")

    temp_file_path = ""
    try:
        # Save the uploaded file temporarily
        file_extension = file.filename.split('.')[-1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name

        extraction_schema = {
            "type": "OBJECT",
            "properties": {
                "vendor_name": {"type": "STRING"},
                "invoice_number": {"type": "STRING"},
                "invoice_date": {"type": "STRING"},
                "category": {"type": "STRING", "description": "Classify as: software, hardware, travel, office, or other"},
                "taxable_value": {"type": "NUMBER"},
                "total_gst": {"type": "NUMBER"},
                "invoice_total": {"type": "NUMBER"},
                "vendor_gstin": {"type": "STRING", "description": "15-digit GSTIN. Return 'MISSING' if not found."},
                "flagged_anomalies": {
                    "type": "ARRAY",
                    "items": {"type": "STRING"},
                    "description": "List mathematical mismatches (taxable_value + total_gst != invoice_total) or missing GSTIN."
                }
            },
            "required": ["vendor_name", "invoice_number", "invoice_date", "category", "taxable_value", "total_gst", "invoice_total"]
        }

        # 2. Resilient API Rotation Loop
        last_error = None
        for key in api_keys:
            try:
                # Initialize client with current key in the pool
                client = genai.Client(api_key=key)
                uploaded_doc = client.files.upload(file=temp_file_path)
                
                response = client.models.generate_content(
                    model='gemini-3.6-flash',
                    contents=[
                        "Extract the accounting data from this financial document accurately.",
                        uploaded_doc
                    ],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=extraction_schema,
                        temperature=0.0
                    )
                )
                
                client.files.delete(name=uploaded_doc.name)
                
                # If successful, return and break the loop instantly
                return json.loads(response.text)
                
            except Exception as e:
                # If the key fails (e.g., Quota Exceeded), capture error and try the next key
                last_error = str(e)
                print(f"API Key {key[:6]}... failed: {last_error}")
                continue 
                
        # If the loop finishes without returning, every single key was exhausted
        raise HTTPException(status_code=429, detail=f"All API keys exhausted. Last error: {last_error}")

    finally:
        # Clean up local temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
