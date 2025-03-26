import os
import openai
import json
import re
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pdf2image import convert_from_bytes
from paddleocr import PaddleOCR
from PIL import Image
import numpy as np
import cv2


app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load OpenRouter API Key
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    raise ValueError("❌ OpenRouter API Key is missing. Set it as an environment variable.")

# Initialize OpenRouter Client
client = openai.OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
)

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang="en", use_gpu=True)

# Create directory for saving enhanced images (optional, for debugging)
OUTPUT_DIR = "./enhanced_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Enhance image before OCR
def enhance_image(image, save_name="enhanced_image"):
    try:
        # Convert PIL Image to OpenCV format
        img = np.array(image)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Increase contrast
        gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=10)

        # Apply adaptive thresholding to binarize
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

        # Convert back to PIL Image
        enhanced_img = Image.fromarray(cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB))

        # Save the enhanced image for inspection (optional)
        save_path = os.path.join(OUTPUT_DIR, f"{save_name}.png")
        enhanced_img.save(save_path)
        print(f"🔹 Saved enhanced image to: {save_path}")

        return enhanced_img
    except Exception as e:
        print(f" Error in image enhancement: {e}")
        return image  # Fallback to original if enhancement fails

# Extract text from images using PaddleOCR
def extract_text_from_image(image, image_id="image"):
    # Enhance the image first
    enhanced_image = enhance_image(image, save_name=image_id)
    enhanced_image = np.array(enhanced_image)  # Convert to NumPy array for PaddleOCR
    result = ocr.ocr(enhanced_image, cls=True)
    extracted_text = " ".join([word_info[1][0] for line in result if line for word_info in line])
    return extracted_text.strip()

# Process extracted text with AI using a dynamic prompt
def process_with_ai(extracted_text, model_prompt=None):
    try:
        # Use provided prompt or fall back to default
        default_prompt = (
            "Convert this text to a JSON array with objects containing key-value pairs "
            "{transaction_code,arp_no,pin,owner,survey_no,lot_no,no_or_street,brgy,municipality,province}. "
            "Return only the JSON array."
        )
        system_prompt = model_prompt if model_prompt else default_prompt

        response = client.chat.completions.create(
            model="x-ai/grok-2-vision-1212",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": extracted_text}
            ],
            temperature=0.2,
            max_tokens=800
        )

        # Debugging: Print Full API Response
        print("Full API Response:", response)

        # Check if response is valid
        if not response or not response.choices or not response.choices[0].message:
            print("AI response is empty or malformed!")
            return {"error": "AI returned an empty response"}

        # Extract AI output and remove Markdown formatting
        structured_data = response.choices[0].message.content.strip()
        structured_data = re.sub(r"```json\n(.*?)\n```", r"\1", structured_data, flags=re.DOTALL).strip()

        # Ensure AI output is valid JSON
        try:
            structured_data = json.loads(structured_data)
        except json.JSONDecodeError:
            print("AI returned invalid JSON!")
            return {"error": "AI returned invalid JSON", "raw_response": structured_data}

        return {"data": structured_data, "prompt_used": system_prompt}

    except Exception as e:
        print(f"OpenRouter API Error: {e}")
        return {"error": str(e)}

# API Endpoint: Upload & Process PDFs/Images with Dynamic Prompt
@app.post("/get-structured-json/")
async def get_structured_json(file: UploadFile = File(...), model_prompt: str = Form(None)):
    try:
        file_bytes = await file.read()
        filename = file.filename.lower()

        extracted_text = ""

        # Check file type
        if filename.endswith(".pdf"):
            images = convert_from_bytes(file_bytes)
            for i, img in enumerate(images):
                image_id = f"{os.path.splitext(filename)[0]}_page_{i+1}"
                extracted_text += extract_text_from_image(img, image_id) + "\n"
        elif filename.endswith((".png", ".jpg", ".jpeg")):
            image = Image.open(file.file)
            image_id = os.path.splitext(filename)[0]
            extracted_text = extract_text_from_image(image, image_id)
        else:
            return {"error": "Unsupported file type. Please upload a PDF, PNG, JPG, or JPEG."}

        # Clean extracted text
        raw_text = extracted_text  # Store raw text before cleaning
        # extracted_text = re.sub(r"[^a-zA-Z0-9\s,.]", "", extracted_text)  # Remove special characters
        # extracted_text = re.sub(r"\s+", " ", extracted_text).strip()  # Normalize spaces
        print("🔹 Cleaned RAW Text:", extracted_text)

        # Process extracted text with AI
        result = process_with_ai(extracted_text, model_prompt)

        # Debugging
        print("🔹 Extracted Result:", result)

        # Check if AI returned valid data
        if "error" in result:
            return {"raw_text": raw_text, "error": result["error"], "prompt_used": result.get("prompt_used", model_prompt)}

        # Return both raw text and structured data
        return {"raw_text": raw_text, "data": result["data"], "prompt_used": result["prompt_used"]}

    except Exception as e:
        print(f"Error: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)