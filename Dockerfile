# CPU inference image for Cloud Run (or any container host).
# Build context must contain models/best_multimodal.pt (train first, or scp it in).
FROM python:3.11-slim

WORKDIR /srv

COPY deploy/requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY config.yml .
COPY src/ src/
COPY app/ app/
COPY models/best_multimodal.pt models/best_multimodal.pt

ENV HF_HOME=/srv/.hf
# Bake the backbone weights into the image so cold starts don't hit the Hub.
RUN python -c "import sys; sys.path.insert(0, '.'); \
    from src.configs import config; \
    from transformers import AutoTokenizer, AutoImageProcessor, AutoModel; \
    AutoTokenizer.from_pretrained(config.model.text_model_name); \
    AutoImageProcessor.from_pretrained(config.model.vision_backbone_name); \
    AutoModel.from_pretrained(config.model.text_model_name); \
    AutoModel.from_pretrained(config.model.vision_backbone_name)"

EXPOSE 8080
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
