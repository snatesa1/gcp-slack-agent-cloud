#!/bin/bash
# check_resources.sh: Audit GCP resources for potential costs

PROJECT_ID=$(gcloud config get-value project)

echo "🔍 Auditing GCP Resources for Project: $PROJECT_ID"
echo "----------------------------------------------------"

# 1. Cloud Run
echo "🚀 Cloud Run Services:"
gcloud run services list --platform managed --format="table(SERVICE,REGION,URL,LAST_DEPLOYED)"

# 2. Artifact Registry (Storage Costs)
echo -e "\n📦 Artifact Registry Repositories (Image Storage):"
gcloud artifacts repositories list --format="table(name,format,location,createTime)"

# 3. Secret Manager
echo -e "\n🔐 Secret Manager Secrets:"
gcloud secrets list --format="table(name,replication.automatic,createTime)"

# 4. Cloud Build (Recent History)
echo -e "\n🏗️ Recent Cloud Builds (Last 5):"
gcloud builds list --limit=5 --format="table(ID,CREATE_TIME,DURATION,STATUS)"

# 5. Compute Engine (Just in case)
echo -e "\n💻 Compute Engine Instances:"
gcloud compute instances list --format="table(name,zone,status)" 2>/dev/null || echo "No Compute instances found or API not enabled."

# 6. Cloud Storage Buckets
echo -e "\n🗄️ Cloud Storage Buckets:"
gcloud storage buckets list --format="table(name,location,storageClass)" 2>/dev/null || echo "No buckets found."

echo -e "\n----------------------------------------------------"
echo "💡 Tip: Artifact Registry and Cloud Storage are the most common sources of idle costs."
echo "💡 Use './destroy_app.sh' to cleanup the primary agent resources."
