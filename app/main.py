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
        use_face=config.model.use_face,
        face_model_name=config.model.face_model_name,
    ).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(config.model.text_model_name)
    feature_extractor = AutoImageProcessor.from_pretrained(config.model.vision_backbone_name)
    print("Models loaded successfully!")
except Exception as e:
    print(f"Warning: Model weights not found or failed to load. Will run dummy predictions. Error: {e}")
    model = None
    load_error = f"{type(e).__name__}: {e}"[:300]
else:
    load_error = None

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

# Direct facial-expression head: the face backbone is a ViT fine-tuned on
# FER2013, but the fusion head above it was trained on MSCTD dialogue-sentiment
# labels, which barely supervise facial expression. For image-only requests
# the FER classifier's own 7-emotion output is the far stronger signal.
fer_model, fer_labels = None, []
if mtcnn is not None:
    try:
        from transformers import AutoModelForImageClassification
        fer_model = AutoModelForImageClassification.from_pretrained(
            config.model.face_model_name).to(device).eval()
        fer_labels = [fer_model.config.id2label[i].lower()
                      for i in range(len(fer_model.config.id2label))]
        print(f"FER head enabled ({', '.join(fer_labels)}).")
    except Exception as e:
        print(f"FER head disabled ({e}).")
        fer_model = None


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
        # Dummy fallback response if models not downloaded yet; surfaces the
        # startup error so remote deployments are diagnosable via the API.
        return {"sentiment": "Neutral", "confidence": 0.99,
                "warning": "Model not loaded properly.", "load_error": load_error}

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
            # scipy ships with the core deps (browsers record 44.1/48 kHz,
            # so this path runs for nearly every mic upload)
            from math import gcd
            from scipy.signal import resample_poly
            g = gcd(int(sr), 16000)
            waveform = resample_poly(waveform, 16000 // g, sr // g)
        waveform_np = waveform.astype("float32")
        audio_values = torch.tensor(waveform_np).unsqueeze(0).to(device)
    else:
        audio_values = torch.zeros((1, 16000)).to(device)

    # MSCTD label encoding (per the dataset README): neutral: 0, negative: 1, positive: 2
    classes = ["Neutral", "Negative", "Positive"]

    def as_pct(p):
        return {c: round(float(p[i]) * 100, 1) for i, c in enumerate(classes)}

    has_real_image = bool(image and image.filename)

    # Prediction (text + image + optional face)
    with torch.no_grad():
        logits = model(input_ids, attention_mask, pixel_values, audio_values, face_values)
        probs = torch.softmax(logits, dim=1)[0]

    result = {}

    # Confidence-modulated late fusion with the speech-tone model: each
    # component's prior is scaled by its certainty (1 - normalized entropy),
    # so a confident voice read outweighs an unsure text read and vice versa.
    if waveform_np is not None and audio_model is not None:
        # Loudness-match the training corpora: their native peaks sit around
        # 0.04-1.0 (median ~0.14) and augmentation only made clips quieter,
        # so anything near digital full scale is louder than the model ever
        # saw and reads as negative prosody. 0.15 scored best on a held-out
        # clip battery; 0.95 flipped clean happy clips to Negative.
        peak = float(abs(waveform_np).max())
        if peak > 1e-6:
            waveform_np = waveform_np * (0.15 / peak)
        feats = audio_fe(waveform_np, sampling_rate=16000,
                         return_tensors="pt")["input_features"].to(device)
        with torch.no_grad():
            audio_probs = torch.softmax(audio_model(feats), dim=1)[0]

        # Conversational-speech calibration: real speech is milder than the
        # acted training corpora, so Neutral dominates in absolute terms even
        # when Neg/Pos are clearly elevated above their baselines.
        # Preferred: a learned affine transform on log-probs fitted from
        # calibration-lab recordings (docs/lab.html -> models/voice_calibration.json).
        # Fallback: hand rule tuned via env knobs.
        raw_audio_probs = audio_probs.clone()
        calib_path = os.getenv("VOICE_CALIBRATION", "models/voice_calibration.json")
        if os.path.exists(calib_path):
            # Per-user transform fitted from a docs/lab.html session recorded
            # through THIS loudness pipeline (owner's fit: 96.7% LOO over 60
            # takes vs 88.3% raw). Delete the file to serve raw model output.
            import json
            calib = json.load(open(calib_path))
            W = torch.tensor(calib["W"], dtype=audio_probs.dtype, device=audio_probs.device)
            b = torch.tensor(calib["b"], dtype=audio_probs.dtype, device=audio_probs.device)
            # eps must match the fitting code (log(p + 1e-4))
            audio_probs = torch.softmax(W @ torch.log(audio_probs + 1e-4) + b, dim=0)
        else:
            # Opt-in only: this hand rule (like the old calibration) was tuned
            # against the pre-fix loudness pipeline. With loudness-matched
            # input the raw model needs no correction.
            cutoff = float(os.getenv("VOICE_NEUTRAL_CUTOFF", "0"))
            damp = float(os.getenv("VOICE_NEUTRAL_DAMP", "0.3"))
            boost = float(os.getenv("VOICE_POSITIVE_BOOST", "2.2"))
            if audio_probs[0] < cutoff:
                audio_probs = audio_probs * torch.tensor([damp, 1.0, boost], device=audio_probs.device)
                audio_probs = audio_probs / audio_probs.sum()

        result["components"] = {
            "text_image": {"label": classes[probs.argmax().item()], "probabilities": as_pct(probs)},
            "voice_tone": {"label": classes[audio_probs.argmax().item()],
                           "probabilities": as_pct(audio_probs),
                           "raw_probabilities": as_pct(raw_audio_probs)},
        }

        if not text.strip() and not has_real_image:
            # Voice-only request: the fusion model would contribute nothing but
            # its blank-input prior (heavily Neutral) - use the voice read alone.
            probs = audio_probs
        else:
            def certainty(p):
                entropy = -(p * (p + 1e-9).log()).sum()
                return float(1 - entropy / torch.log(torch.tensor(float(len(p)))))

            prior_mm = 0.6 if text.strip() else 0.35   # voice leads when there is no text
            w_mm = prior_mm * certainty(probs)
            w_audio = (1 - prior_mm) * certainty(audio_probs)
            probs = (w_mm * probs + w_audio * audio_probs) / (w_mm + w_audio + 1e-9)

    # Facial-expression read on the detected face crop, via the FER head.
    # Image-only requests use it as the verdict (personal calibration from
    # docs/face-lab.html -> models/face_calibration.json when available,
    # otherwise a fixed emotion->valence mapping); requests with text keep
    # the fusion verdict and surface the expression as a component.
    if face_values is not None and fer_model is not None:
        with torch.no_grad():
            fer_probs = torch.softmax(fer_model(pixel_values=face_values).logits, dim=1)[0]
        result["face_expression"] = {lab: round(float(p) * 100, 1)
                                     for lab, p in zip(fer_labels, fer_probs)}
        if not text.strip():
            result["raw_probabilities"] = as_pct(probs)  # fusion read, for reference
            calib = None
            fc_path = os.getenv("FACE_CALIBRATION", "models/face_calibration.json")
            if os.path.exists(fc_path):
                import json
                fc = json.load(open(fc_path))
                if fc.get("type") == "fer7":
                    calib = fc
            if calib is not None:
                order = [fer_labels.index(lab) for lab in calib["labels"]]
                x = torch.log(fer_probs[order] + 1e-4)
                W = torch.tensor(calib["W"], dtype=x.dtype, device=x.device)
                b = torch.tensor(calib["b"], dtype=x.dtype, device=x.device)
                probs = torch.softmax(W @ x + b, dim=0)
            else:
                g = {lab: float(p) for lab, p in zip(fer_labels, fer_probs)}
                mapped = torch.tensor(
                    [g.get("neutral", 0.0),
                     g.get("angry", 0.0) + g.get("disgust", 0.0)
                     + g.get("fear", 0.0) + g.get("sad", 0.0),
                     g.get("happy", 0.0) + g.get("surprise", 0.0)],
                    device=probs.device)
                probs = mapped / mapped.sum()

    if has_real_image and mtcnn is not None:
        result["face_detected"] = face_values is not None
    elif has_real_image and config.model.use_face and mtcnn is None:
        result["face_note"] = ("face detector not installed on this server - "
                               "pip install --no-deps facenet-pytorch and restart")

    pred_idx = probs.argmax().item()
    result.update({"sentiment": classes[pred_idx],
                   "confidence": round(probs[pred_idx].item(), 4),
                   "probabilities": as_pct(probs)})
    return result

if __name__ == "__main__":
    import uvicorn
    # Bind to localhost by default; opt into wider exposure via HOST explicitly.
    uvicorn.run(app, host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "8000")))
