from pydantic_settings import BaseSettings
from google.cloud import secretmanager
import os
import sys

import google.auth

class Settings(BaseSettings):
    PROJECT_ID: str = ""
    ALPACA_API_KEY: str = ""
    ALPACA_SECRET_KEY: str = ""
    SLACK_BOT_TOKEN: str = ""
    SLACK_SIGNING_SECRET: str = ""
    VERTEX_LOCATION: str = "asia-southeast1"
    VERTEX_MODEL: str = "gemini-3-flash-preview"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Auto-detect project ID if not set
        if not self.PROJECT_ID:
            try:
                _, project_id = google.auth.default()
                self.PROJECT_ID = project_id or os.getenv("GOOGLE_CLOUD_PROJECT", "default-project")
            except Exception:
                self.PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "default-project")

    def load_secrets(self):
        """Loads secrets from Google Secret Manager."""
        # In production, we ALWAYS load from Secret Manager
        try:
            client = secretmanager.SecretManagerServiceClient()
            
            def get_secret(secret_id):
                name = f"projects/{self.PROJECT_ID}/secrets/{secret_id}/versions/latest"
                response = client.access_secret_version(request={"name": name})
                return response.payload.data.decode("UTF-8")

            self.ALPACA_API_KEY = get_secret("ALPACA_API_KEY")
            self.ALPACA_SECRET_KEY = get_secret("ALPACA_SECRET_KEY")
            self.SLACK_BOT_TOKEN = get_secret("SLACK_BOT_TOKEN")
            self.SLACK_SIGNING_SECRET = get_secret("SLACK_SIGNING_SECRET")
            print("✓ Successfully loaded production secrets from Secret Manager")
        except Exception as e:
            print(f"❌ Critical Error loading secrets from Secret Manager: {e}")
            # For cloud deployment, this is fatal
            if os.getenv("K_SERVICE"):
                sys.exit(1)
            else:
                # Fallback to env vars for development/testing if not in Cloud Run
                self.ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
                self.ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
                self.SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN", "")
                self.SLACK_SIGNING_SECRET = os.getenv("SLACK_SIGNING_SECRET", "")
                print("⚠️ Falling back to environment variables for secrets")

settings = Settings()
settings.load_secrets()
