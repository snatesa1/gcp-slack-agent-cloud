#!/bin/bash
# deploy_app.sh: Upload secrets and deploy to Cloud Run

SERVICE_NAME="slack-trading-agent"
REGION="asia-southeast1"

# Load .env
if [ -f .env ]; then
    export $(cat .env | grep -v '#' | awk '/=/ {print $1}')
fi

PROJECT_ID=$(gcloud config get-value project)
PROJECT_NUMBER=$(gcloud projects describe $PROJECT_ID --format="value(projectNumber)")
SA_EMAIL="$PROJECT_NUMBER-compute@developer.gserviceaccount.com"

echo "🛰️ Project: $PROJECT_ID ($PROJECT_NUMBER)"

# 1. Upload Secrets (Automated)
echo "🔐 Syncing secrets to Secret Manager..."
chmod +x upload_secrets.sh
./upload_secrets.sh

# 2. Grant Permissions
echo "🔑 Granting IAM permissions to service account..."
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:$SA_EMAIL" \
  --role="roles/secretmanager.secretAccessor"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:$SA_EMAIL" \
  --role="roles/aiplatform.user"

# 3. Deploy to Cloud Run
echo "🚢 Deploying to Cloud Run (4GiB Memory, asia-southeast1)..."
gcloud run deploy $SERVICE_NAME \
  --source . \
  --region $REGION \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 2 \
  --timeout 600 \
  --set-env-vars GOOGLE_CLOUD_PROJECT=$PROJECT_ID \
  --service-account $SA_EMAIL

echo "✅ Deployment complete!"
