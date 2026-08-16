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
from PIL import Image
from transformers import AutoTokenizer, AutoImageProcessor
from src.models.multimodal import MultimodalFusionNet
from src.configs import config

MODEL_PATH = os.getenv("MODEL_PATH", "models/best_multimodal.pt")

app = FastAPI(title="Multimodal Sentiment Analysis")

# Allow the static playground (docs/ on GitHub Pages, or a local file) to call
# this API from another origin. Demo-friendly; tighten for real deployments.
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return HTMLResponse("")


app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Loading models (this might take a few seconds)...")

try:
    model = MultimodalFusionNet(
        text_model_name=config.model.text_model_name,
        vit_model_name=config.model.vision_backbone_name,
        audio_model_name=config.model.audio_model_name,
        use_audio=config.model.use_audio,
    ).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(config.model.text_model_name)
    feature_extractor = AutoImageProcessor.from_pretrained(config.model.vision_backbone_name)
    print("Models loaded successfully!")
except Exception as e:
    print(f"Warning: Model weights not found or failed to load. Will run dummy predictions. Error: {e}")
    model = None

# Optional face-expression branch: detect the largest face at inference and
# feed the crop to the fusion model (mirrors training-time precomputed crops).
face_processor, mtcnn = None, None
if model is not None and config.model.use_face:
    try:
        from facenet_pytorch import MTCNN
        face_processor = AutoImageProcessor.from_pretrained(config.model.face_model_name)
        mtcnn = MTCNN(select_largest=True, post_process=False, device=str(device))
        print("Face branch enabled.")
    except Exception as e:
        print(f"Face branch disabled ({e}).")
        mtcnn = None


def detect_face_crop(pil_image):
    """Largest-face crop with 20% context padding, or None."""
    if mtcnn is None:
        return None
    try:
        boxes, probs = mtcnn.detect(pil_image)
        if boxes is None or probs[0] is None or probs[0] < 0.90:
            return None
        x1, y1, x2, y2 = boxes[0]
        w, h = x2 - x1, y2 - y1
        x1, y1 = max(0, x1 - 0.2 * w), max(0, y1 - 0.2 * h)
        x2 = min(pil_image.width, x2 + 0.2 * w)
        y2 = min(pil_image.height, y2 + 0.2 * h)
        return pil_image.crop((x1, y1, x2, y2)).resize((224, 224))
    except Exception:
        return None


# Optional speech-tone model (late fusion). Missing checkpoint just disables it.
AUDIO_MODEL_PATH = os.getenv("AUDIO_MODEL_PATH", "models/audio_sentiment.pt")
audio_model, audio_fe = None, None
try:
    from transformers import WhisperFeatureExtractor
    from src.models.audio_sentiment import AudioSentimentModel
    audio_model = AudioSentimentModel().to(device)
    audio_model.load_state_dict(torch.load(AUDIO_MODEL_PATH, map_location=device))
    audio_model.eval()
    audio_fe = WhisperFeatureExtractor.from_pretrained("openai/whisper-base")
    print("Speech-tone model loaded.")
except Exception as e:
    print(f"Speech-tone model disabled ({e}).")
    audio_model = None

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict_sentiment(
    text: str = Form(""),
    image: UploadFile = File(None),
    audio: UploadFile = File(None)
):
    if not model:
        # Dummy fallback response if models not downloaded yet
        return {"sentiment": "Neutral", "confidence": 0.99, "warning": "Model not loaded properly."}

    # 1. Process Text
    inputs = tokenizer(text, return_tensors="pt",
                       max_length=config.model.max_text_len, truncation=True)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # 2. Process Image (+ optional face crop for the expression branch)
    face_values = None
    if image and image.filename:
        img_bytes = await image.read()
        pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        pixel_values = feature_extractor(images=pil_image, return_tensors="pt")["pixel_values"].to(device)
        face_crop = detect_face_crop(pil_image)
        if face_crop is not None and face_processor is not None:
            face_values = face_processor(images=face_crop, return_tensors="pt")["pixel_values"].to(device)
    else:
        # Blank image through the processor — matches the training-time
        # missing-image fallback distribution (raw zeros would not).
        blank = Image.new("RGB", (224, 224))
        pixel_values = feature_extractor(images=blank, return_tensors="pt")["pixel_values"].to(device)

    # 3. Process Audio
    waveform_np = None
    if audio and audio.filename:
        aud_bytes = await audio.read()
        import soundfile as sf
        waveform, sr = sf.read(io.BytesIO(aud_bytes))
        # Ensure mono and 16k hr
        if len(waveform.shape) > 1:
            waveform = waveform.mean(axis=1)
        if sr != 16000:
            import librosa  # optional dep; only needed to resample non-16kHz uploads
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=16000)
        waveform_np = waveform.astype("float32")
        audio_values = torch.tensor(waveform_np).unsqueeze(0).to(device)
    else:
        audio_values = torch.zeros((1, 16000)).to(device)

    # MSCTD label encoding (per the dataset README): neutral: 0, negative: 1, positive: 2
    classes = ["Neutral", "Negative", "Positive"]

    # Prediction (text + image + optional face)
    with torch.no_grad():
        logits = model(input_ids, attention_mask, pixel_values, audio_values, face_values)
        probs = torch.softmax(logits, dim=1)[0]

    result = {}

    # Confidence-modulated late fusion with the speech-tone model: each
    # component's prior is scaled by its certainty (1 - normalized entropy),
    # so a confident voice read outweighs an unsure text read and vice versa.
    if waveform_np is not None and audio_model is not None:
        # Peak-normalize: consumer-mic recordings are far quieter than the
        # loudness-normalized studio clips the model trained on.
        peak = float(abs(waveform_np).max())
        if peak > 1e-6:
            waveform_np = waveform_np * (0.95 / peak)
        feats = audio_fe(waveform_np, sampling_rate=16000,
                         return_tensors="pt")["input_features"].to(device)
        with torch.no_grad():
            audio_probs = torch.softmax(audio_model(feats), dim=1)[0]

        def certainty(p):
            entropy = -(p * (p + 1e-9).log()).sum()
            return float(1 - entropy / torch.log(torch.tensor(float(len(p)))))

        prior_mm = 0.6 if text.strip() else 0.35   # voice leads when there is no text
        w_mm = prior_mm * certainty(probs)
        w_audio = (1 - prior_mm) * certainty(audio_probs)
        total = w_mm + w_audio + 1e-9
        result["components"] = {
            "text_image": classes[probs.argmax().item()],
            "voice_tone": classes[audio_probs.argmax().item()],
        }
        probs = (w_mm * probs + w_audio * audio_probs) / total

    pred_idx = probs.argmax().item()
    result.update({"sentiment": classes[pred_idx],
                   "confidence": round(probs[pred_idx].item(), 4)})
    return result

if __name__ == "__main__":
    import uvicorn
    # Bind to localhost by default; opt into wider exposure via HOST explicitly.
    uvicorn.run(app, host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "8000")))
