import os
import openai
import json
import re
import httpx  # Add this import
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pdf2image import convert_from_bytes
from PIL import Image, ImageEnhance
from io import BytesIO
import base64
import cv2
import numpy as np
import imagehash


app = FastAPI()

# CORS middleware for Netlify frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://ramesesdocumentprocessor.netlify.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
print("🔹 CORS middleware enabled for https://ramesesdocumentprocessor.netlify.app")

# Load OpenRouter API Key
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("❌ OpenRouter API Key is missing.")

# Initialize OpenRouter Client with explicit httpx client
client = openai.OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
    http_client=httpx.Client()  # Explicitly use httpx.Client without proxies
)

# Preprocess image (no save to disk for Render)
def preprocess_image(image):
    try:
        img = np.array(image)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        sharpening_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        gray = cv2.filter2D(gray, -1, sharpening_kernel)
        gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=10)
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 5
        )
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        processed_img = Image.fromarray(cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB))
        enhancer = ImageEnhance.Contrast(processed_img)
        return enhancer.enhance(1.2)
    except Exception as e:
        print(f"🔹 Error in preprocessing: {e}")
        return image

# Extract and process text with Grok Vision
def extract_and_process_text(image, image_id="image", model_prompt=None):
    try:
        original_image = Image.fromarray(np.array(image))
        original_hash = imagehash.average_hash(original_image)
        image = preprocess_image(image)
        processed_hash = imagehash.average_hash(image)
        print(f"🔹 Original Image Hash: {original_hash}")
        print(f"🔹 Processed Image Hash: {processed_hash}")
        if original_hash == processed_hash:
            print("🔹 Warning: Preprocessing did not alter the image!")
        else:
            print("🔹 Preprocessing successfully altered the image.")

        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        default_prompt = (
            "Extract all text from the image as raw text. Then, convert this raw text to a JSON array with objects "
            "containing {name,age,status}. "
            "Return the response in this format: ```raw\n<raw_text>\n```\n```json\n<json_array>\n```"
        )
        system_prompt = model_prompt if model_prompt else default_prompt

        print(f"🔹 Sending to Grok Vision with prompt: {system_prompt}")
        response = client.chat.completions.create(
            model="x-ai/grok-2-vision-1212",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_str}"}}]}
            ],
            temperature=0.2,
            max_tokens=800
        )

        result = response.choices[0].message.content.strip()
        print(f"🔹 Raw Grok Vision Response: {result}")

        # Parse response
        try:
            raw_text_match = re.search(r"```raw\n(.*?)\n```", result, re.DOTALL)
            json_match = re.search(r"```json\n(.*?)\n```", result, re.DOTALL)
            raw_text = raw_text_match.group(1).strip() if raw_text_match else "No raw text extracted"
            json_str = json_match.group(1).strip() if json_match else result
            data = json.loads(json_str)
            return {"raw_text": raw_text, "data": data, "prompt_used": system_prompt}
        except (ValueError, json.JSONDecodeError) as e:
            print(f"🔹 Failed to parse response: {e}")
            return {"raw_response": result, "prompt_used": system_prompt}
    except Exception as e:
        print(f"🔹 Error in Grok Vision processing: {e}")
        return {"error": str(e)}

@app.post("/get-structured-json/")
async def get_structured_json(file: UploadFile = File(...), model_prompt: str = Form(None)):
    print("🔹 Received POST request")
    try:
        file_bytes = await file.read()
        if len(file_bytes) > 2 * 1024 * 1024:  # 2MB limit for Render free tier
            return JSONResponse(content={"error": "File too large. Max size is 2MB."}, status_code=400)

        filename = file.filename.lower()
        if not filename.endswith((".pdf", ".png", ".jpg", ".jpeg")):
            return JSONResponse(content={"error": "Unsupported file type."}, status_code=400)

        results = {"raw_text": "", "data": [], "prompt_used": ""}
        if filename.endswith(".pdf"):
            images = convert_from_bytes(file_bytes, dpi=300)
            for i, img in enumerate(images):
                image_id = f"{os.path.splitext(filename)[0]}_page_{i+1}"
                result = extract_and_process_text(img, image_id, model_prompt)
                if "error" in result:
                    return JSONResponse(content=result, status_code=500)
                if "raw_response" in result:
                    return JSONResponse(content=result)
                results["raw_text"] += result["raw_text"] + "\n"
                results["data"].extend(result["data"])
                results["prompt_used"] = result["prompt_used"]
        else:
            image = Image.open(BytesIO(file_bytes))
            image_id = os.path.splitext(filename)[0]
            result = extract_and_process_text(image, image_id, model_prompt)
            if "error" in result:
                return JSONResponse(content=result, status_code=500)
            if "raw_response" in result:
                return JSONResponse(content=result)
            results["raw_text"] = result["raw_text"]
            results["data"] = result["data"]
            results["prompt_used"] = result["prompt_used"]

        print("🔹 Extracted Results:", results)
        return JSONResponse(content=results)
    except Exception as e:
        print(f"🔹 Error: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)