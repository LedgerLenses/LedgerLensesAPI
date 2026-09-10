import os
import json
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types

app = FastAPI(title="LedgerLenses API")

# CORS setup - Vital for decoupled Cloudflare UI -> Render Backend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this to your .pages.dev URL later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Gemini Client (Pulls GEMINI_API_KEY from Render environment automatically)
client = genai.Client()

@app.get("/health")
async def health_check():
    """The cold-start ping endpoint to wake the server up instantly."""
    return {"status": "hot", "system": "LedgerLenses Backend Running"}

@app.post("/api/extract")
async def extract_document(file: UploadFile = File(...)):
    """Handles PDF and Image uploads, processes via Gemini, returns strict JSON."""
    
    if file.content_type not in ["application/pdf", "image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Unsupported file format. Please upload PDF, JPG, or PNG.")

    temp_file_path = ""
    try:
        # 1. Save uploaded file temporarily so the Gemini API can read it
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.filename.split('.')[-1]}") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name

        # 2. Upload to Google AI Studio (Required for processing PDFs)
        uploaded_doc = client.files.upload(file=temp_file_path, mime_type=file.content_type)

        # 3. Define the strict JSON Schema for the Indian FinTech context
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

        # 4. Call the multimodal engine
        response = client.models.generate_content(
            model='gemini-3.6-flash',
            contents=[
                "Extract the accounting data from this financial document accurately.",
                uploaded_doc
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=extraction_schema,
                temperature=0.0 # Force absolute determinism, no creative hallucination
            )
        )

        # 5. Clean up Gemini server storage to avoid quota limits
        client.files.delete(name=uploaded_doc.name)

        # 6. Parse and return the JSON
        return json.loads(response.text)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up local temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
