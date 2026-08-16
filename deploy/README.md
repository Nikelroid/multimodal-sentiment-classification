# Deploying the inference API to Google Cloud Run

CPU-only Cloud Run is the right shape for this demo: it scales to zero (you pay
only per request), a prediction takes a couple of seconds on 2 vCPUs, and demo
traffic costs cents — a ~$50 credit effectively never runs out. A GPU or an
always-on instance would burn the credit in weeks; don't use one for this.

## 1. Get the trained checkpoint onto your laptop

```bash
# from the repo root on your laptop
scp <user>@discovery.usc.edu:/home1/kelidari/multimodal-sentiment-classification/models/best_multimodal.pt models/
```

## 2. Deploy (one command)

Requires the gcloud CLI, logged in (`gcloud auth login`) with a project selected
(`gcloud config set project <PROJECT_ID>`).

```bash
gcloud run deploy multimodal-sentiment \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi --cpu 2 \
  --min-instances 0 --max-instances 2
```

Cloud Build uses the repo `Dockerfile`; `.gcloudignore` makes sure the
checkpoint is uploaded with the build context. First build takes ~10 min
(it bakes the RoBERTa/ViT backbones into the image so cold starts skip the
Hugging Face Hub).

## 3. Point the playground at it

The deploy prints a service URL like
`https://multimodal-sentiment-xxxxxxxx-uc.a.run.app`. Smoke-test it:

```bash
curl -s -X POST -F "text=I love this!" https://<your-service-url>/predict
```

Then set `DEFAULT_API` (one line at the top of the `<script>` block in
`docs/index.html`) to that URL, commit, and push — the playground on
GitHub Pages now uses the cloud model by default, and visitors can still
switch the endpoint field to `http://localhost:8000` to use their own.

## Notes

- Keep `--min-instances 0`. The trade-off is a ~20–40 s cold start after idle
  periods; a warm instance would cost roughly $15–30/month of your credit.
- The API is public (`--allow-unauthenticated`) with permissive CORS — fine
  for a demo endpoint that serves only predictions.
- Local serving works on a MacBook without CUDA: `pip install -r
  deploy/requirements-serve.txt` gives you the CPU build of torch, then
  `uvicorn app.main:app --port 8000`.
