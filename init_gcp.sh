#!/bin/bash
# init_gcp.sh: Initial setup for GCP project

PROJECT_ID="sonic-terrain-485512-e0"

echo "🔐 Authenticating with Google Cloud..."
gcloud auth login

echo "🎯 Setting project to: $PROJECT_ID"
gcloud config set project $PROJECT_ID

echo "🚀 Enabling required APIs..."
gcloud services enable \
  aiplatform.googleapis.com \
  secretmanager.googleapis.com \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  iam.googleapis.com

echo "✅ Initialization complete!"
