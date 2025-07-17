# Despliegue en Google Cloud Run

1. **Inicializar gcloud**

```bash
gcloud init

gcloud services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  containerregistry.googleapis.com

gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/lead-evaluator .

gcloud run deploy lead-evaluator \
  --image gcr.io/reflected-jet-466200-a4/lead-evaluator \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
