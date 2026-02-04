#!/bin/bash

# Script to upload secrets from .env to GCP Secret Manager
# Usage: ./upload_secrets.sh

# Load .env file
if [ -f .env ]; then
    export $(cat .env | grep -v '#' | awk '/=/ {print $1}')
fi

PROJECT_ID=$(gcloud config get-value project)

echo "🛰️ Uploading secrets to project: $PROJECT_ID"

create_and_upload_secret() {
    SECRET_NAME=$1
    SECRET_VALUE=$2

    if [ -z "$SECRET_VALUE" ]; then
        echo "⚠️ $SECRET_NAME is empty in .env, skipping..."
        return
    fi

    # Check if secret exists
    gcloud secrets describe $SECRET_NAME --project=$PROJECT_ID > /dev/null 2>&1
    if [ $? -ne 0 ]; then
        echo "🆕 Creating secret: $SECRET_NAME"
        gcloud secrets create $SECRET_NAME --replication-policy="automatic" --project=$PROJECT_ID
    fi

    echo "⬆️ Uploading version for: $SECRET_NAME"
    echo -n "$SECRET_VALUE" | gcloud secrets versions add $SECRET_NAME --data-file=- --project=$PROJECT_ID
}

create_and_upload_secret "ALPACA_API_KEY" "$ALPACA_API_KEY"
create_and_upload_secret "ALPACA_SECRET_KEY" "$ALPACA_SECRET_KEY"
create_and_upload_secret "SLACK_BOT_TOKEN" "$SLACK_BOT_TOKEN"
create_and_upload_secret "SLACK_SIGNING_SECRET" "$SLACK_SIGNING_SECRET"

echo "✅ All secrets uploaded to Secret Manager!"
