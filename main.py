import os
import openai
import json
import signal
import sys
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pdf2image import convert_from_bytes
from paddleocr import PaddleOCR
from PIL import Image
import numpy as np
import cv2

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

# Lazy PaddleOCR initialization
ocr = None
def get_ocr():
    global ocr
    if ocr is None:
        print("🔹 Initializing PaddleOCR...")
        ocr = PaddleOCR(use_angle_cls=True, lang="en", use_gpu=False)
    return ocr

# Enhance image
def enhance_image(image):
    try:
        img = np.array(image)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=10)
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        return Image.fromarray(cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB))
    except Exception as e:
        print(f"Error in image enhancement: {e}")
        return image

# Extract text
def extract_text_from_image(image, image_id="image"):
    print(f"🔹 Processing image: {image_id}")
    enhanced_image = enhance_image(image)
    enhanced_image = np.array(enhanced_image)
    ocr_instance = get_ocr()
    result = ocr_instance.ocr(enhanced_image, cls=True)
    return " ".join([word_info[1][0] for line in result if line for word_info in line]).strip()

# Process with AI
def process_with_ai(extracted_text, model_prompt=None):
    default_prompt = (
        "Convert this text to a JSON array with objects containing key-value pairs "
        "{transaction_code,arp_no,pin,owner,survey_no,lot_no,no_or_street,brgy,municipality,province}. "
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
        structured_data = response.choices[0].message.content.strip()
        structured_data = response.sub(r"```json\n(.*?)\n```", r"\1", structured_data, flags=response.DOTALL).strip()
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