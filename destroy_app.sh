#!/bin/bash
# destroy_app.sh: Cleanup all created resources

SERVICE_NAME="slack-trading-agent"
REGION="asia-southeast1"

echo "⚠️  WARNING: This will delete the Cloud Run service and all associated secrets!"
read -p "Are you sure? (y/N): " confirm
if [[ $confirm != [yY] ]]; then
    echo "Deletion cancelled."
    exit 1
fi

PROJECT_ID=$(gcloud config get-value project)

# 1. Delete Cloud Run Service
echo "🗑️ Deleting Cloud Run service: $SERVICE_NAME..."
gcloud run services delete $SERVICE_NAME --region $REGION --quiet

# 2. Delete Secrets
echo "🗑️ Deleting secrets from Secret Manager..."
secrets=("ALPACA_API_KEY" "ALPACA_SECRET_KEY" "SLACK_BOT_TOKEN" "SLACK_SIGNING_SECRET")
for secret in "${secrets[@]}"; do
    gcloud secrets delete $secret --project=$PROJECT_ID --quiet 2>/dev/null
done

echo "✅ Resources destroyed successfully."
