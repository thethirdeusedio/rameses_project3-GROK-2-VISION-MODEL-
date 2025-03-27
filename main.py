import os
import openai
import json
import signal
import sys
import psutil
import pytesseract
import re
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pdf2image import convert_from_bytes
from PIL import Image


app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://ramesesdocumentprocessor.netlify.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
print("🔹 CORS middleware enabled for https://ramesesdocumentprocessor.netlify.app")

# SIGTERM handler
def signal_handler(sig, frame):
    print("🔹 Received SIGTERM, shutting down gracefully")
    sys.exit(0)

signal.signal(signal.SIGTERM, signal_handler)

# Load OpenRouter API Key
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("❌ OpenRouter API Key is missing.")

# Initialize OpenRouter Client
client = openai.OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
)

# Dummy text extraction (bypass OCR)
def extract_text_from_image(image, image_id="image"):
    print(f"🔹 Processing image: {image_id}")
    print(f"🔹 Memory before processing: {psutil.virtual_memory().percent}%")
    text = pytesseract.image_to_string(image)
    print(f"🔹 Memory after processing: {psutil.virtual_memory().percent}%")
    return text.strip()

# Process with AI
def process_with_ai(extracted_text, model_prompt=None):
    default_prompt = (
        "Convert this text to a JSON array with objects containing key-value pairs "
        "{transact_code,arp_no,pin,owner}. "
        "Return only the JSON array."
    )
    system_prompt = model_prompt if model_prompt else default_prompt
    try:
        print("🔹 Sending to OpenRouter...")
        response = client.chat.completions.create(
            model="x-ai/grok-2-vision-1212",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": extracted_text}
            ],
            temperature=0.2,
            max_tokens=800
        )
        # Log raw response for debugging
        raw_content = response.choices[0].message.content
        print(f"🔹 Raw OpenRouter Response: {raw_content}")
        
        # Clean and parse
        structured_data = raw_content.strip()
        structured_data = re.sub(r"```json\n(.*?)\n```", r"\1", structured_data, flags=re.DOTALL).strip()
        structured_data = json.loads(structured_data)
        print("🔹 AI processing complete")
        return {"data": structured_data, "prompt_used": system_prompt}
    except Exception as e:
        print(f"OpenRouter API Error: {e}")
        return {"error": str(e)}

@app.get("/health")
async def health_check():
    return JSONResponse(
        content={"status": "ok"},
        headers={"Access-Control-Allow-Origin": "https://ramesesdocumentprocessor.netlify.app"}
    )

@app.options("/get-structured-json/")
async def options_handler():
    return JSONResponse(
        content={},
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": "https://ramesesdocumentprocessor.netlify.app",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "*"
        }
    )

@app.post("/get-structured-json/")
async def get_structured_json(file: UploadFile = File(...), model_prompt: str = Form(None)):
    print("🔹 Received POST request")
    try:
        print("🔹 Reading file...")
        file_bytes = await file.read()
        if len(file_bytes) > 2 * 1024 * 1024:  # 2MB limit
            print("🔹 File too large")
            return JSONResponse(
                content={"error": "File too large. Max size is 2MB."},
                headers={"Access-Control-Allow-Origin": "https://ramesesdocumentprocessor.netlify.app"},
                status_code=400
            )
        print("🔹 File read successfully")
        filename = file.filename.lower()
        extracted_text = ""

        if filename.endswith(".pdf"):
            print("🔹 Converting PDF...")
            images = convert_from_bytes(file_bytes)[:1]
            for i, img in enumerate(images):
                image_id = f"{os.path.splitext(filename)[0]}_page_{i+1}"
                extracted_text += extract_text_from_image(img, image_id) + "\n"
        elif filename.endswith((".png", ".jpg", ".jpeg")):
            print("🔹 Processing image file...")
            image = Image.open(file.file)
            image_id = os.path.splitext(filename)[0]
            extracted_text = extract_text_from_image(image, image_id)
        else:
            print("🔹 Unsupported file type")
            return JSONResponse(
                content={"error": "Unsupported file type. Please upload a PDF, PNG, JPG, or JPEG."},
                headers={"Access-Control-Allow-Origin": "https://ramesesdocumentprocessor.netlify.app"},
                status_code=400
            )

        raw_text = extracted_text.strip()
        print("🔹 Raw Text:", raw_text)

        result = process_with_ai(raw_text, model_prompt)
        print("🔹 Extracted Result:", result)

        if "error" in result:
            return JSONResponse(
                content={"raw_text": raw_text, "error": result["error"], "prompt_used": result.get("prompt_used", model_prompt)},
                headers={"Access-Control-Allow-Origin": "https://ramesesdocumentprocessor.netlify.app"}
            )

        return JSONResponse(
            content={"raw_text": raw_text, "data": result["data"], "prompt_used": result["prompt_used"]},
            headers={"Access-Control-Allow-Origin": "https://ramesesdocumentprocessor.netlify.app"}
        )

    except Exception as e:
        print(f"🔹 Error in processing: {e}")
        return JSONResponse(
            content={"error": f"Server error: {str(e)}"},
            status_code=500,
            headers={"Access-Control-Allow-Origin": "https://ramesesdocumentprocessor.netlify.app"}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)