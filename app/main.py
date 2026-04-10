import os
import sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Add project root to path so 'src' is found
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import torch
import io
import librosa
from PIL import Image
from transformers import AutoTokenizer, AutoImageProcessor
from src.models.multimodal import MultimodalFusionNet
from src.config import TEXT_MODEL_NAME, VISION_BACKBONE_NAME

app = FastAPI(title="Multimodal Sentiment Analysis")

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return HTMLResponse("")


app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Loading models (this might take a few seconds)...")

try:
    model = MultimodalFusionNet(use_audio=True).to(device)
    # model.load_state_dict(torch.load("models/best_multimodal.pt", map_location=device))
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
    feature_extractor = AutoImageProcessor.from_pretrained(VISION_BACKBONE_NAME)
    print("Models loaded successfully!")
except Exception as e:
    print(f"Warning: Model weights not found or failed to load. Will run dummy predictions. Error: {e}")
    model = None

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict_sentiment(
    text: str = Form(...),
    image: UploadFile = File(None),
    audio: UploadFile = File(None)
):
    if not model:
        # Dummy fallback response if models not downloaded yet
        return {"sentiment": "Neutral", "confidence": 0.99, "warning": "Model not loaded properly."}

    # 1. Process Text
    inputs = tokenizer(text, return_tensors="pt", max_length=50, truncation=True)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    # 2. Process Image
    if image and image.filename:
        img_bytes = await image.read()
        pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        pixel_values = feature_extractor(images=pil_image, return_tensors="pt")["pixel_values"].to(device)
    else:
        # Dummy image
        pixel_values = torch.zeros((1, 3, 224, 224)).to(device)

    # 3. Process Audio
    if audio and audio.filename:
        aud_bytes = await audio.read()
        import soundfile as sf
        waveform, sr = sf.read(io.BytesIO(aud_bytes))
        # Ensure mono and 16k hr
        if len(waveform.shape) > 1:
            waveform = waveform.mean(axis=1)
        if sr != 16000:
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=16000)
        audio_values = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0).to(device)
    else:
        audio_values = torch.zeros((1, 16000)).to(device)

    # Prediction
    with torch.no_grad():
        logits = model(input_ids, attention_mask, pixel_values, audio_values)
        probs = torch.softmax(logits, dim=1)
        pred_idx = probs.argmax(dim=1).item()
        confidence = probs[0, pred_idx].item()
        
    classes = ["Negative", "Neutral", "Positive"]
    
    return {"sentiment": classes[pred_idx], "confidence": round(confidence, 4)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
