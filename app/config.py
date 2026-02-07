from pydantic_settings import BaseSettings
from google.cloud import secretmanager
import os
import sys

import google.auth

class Settings(BaseSettings):
    PROJECT_ID: str = ""
    _secrets: dict = {}
    
    # Internal secret IDs
    _SECRET_IDS = [
        "ALPACA_API_KEY",
        "ALPACA_SECRET_KEY",
        "SLACK_BOT_TOKEN",
        "SLACK_SIGNING_SECRET",
        "FRED_API_KEY"
    ]
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Auto-detect project ID if not set
        if not self.PROJECT_ID:
            try:
                _, project_id = google.auth.default()
                self.PROJECT_ID = project_id or os.getenv("GOOGLE_CLOUD_PROJECT", "default-project")
            except Exception:
                self.PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "default-project")

    def _get_secret_manager_value(self, secret_id: str) -> str:
        """Helper to fetch a single secret from Google Secret Manager."""
        try:
            client = secretmanager.SecretManagerServiceClient()
            name = f"projects/{self.PROJECT_ID}/secrets/{secret_id}/versions/latest"
            response = client.access_secret_version(request={"name": name})
            return response.payload.data.decode("UTF-8")
        except Exception as e:
            print(f"❌ Error loading secret {secret_id} from Secret Manager: {e}")
            return os.getenv(secret_id, "")

    @property
    def ALPACA_API_KEY(self) -> str:
        if "ALPACA_API_KEY" not in self._secrets:
            self._secrets["ALPACA_API_KEY"] = self._get_secret_manager_value("ALPACA_API_KEY")
        return self._secrets["ALPACA_API_KEY"]

    @property
    def ALPACA_SECRET_KEY(self) -> str:
        if "ALPACA_SECRET_KEY" not in self._secrets:
            self._secrets["ALPACA_SECRET_KEY"] = self._get_secret_manager_value("ALPACA_SECRET_KEY")
        return self._secrets["ALPACA_SECRET_KEY"]

    @property
    def SLACK_BOT_TOKEN(self) -> str:
        if "SLACK_BOT_TOKEN" not in self._secrets:
            self._secrets["SLACK_BOT_TOKEN"] = self._get_secret_manager_value("SLACK_BOT_TOKEN")
        return self._secrets["SLACK_BOT_TOKEN"]

    @property
    def SLACK_SIGNING_SECRET(self) -> str:
        if "SLACK_SIGNING_SECRET" not in self._secrets:
            self._secrets["SLACK_SIGNING_SECRET"] = self._get_secret_manager_value("SLACK_SIGNING_SECRET")
        return self._secrets["SLACK_SIGNING_SECRET"]

    @property
    def FRED_API_KEY(self) -> str:
        if "FRED_API_KEY" not in self._secrets:
            self._secrets["FRED_API_KEY"] = self._get_secret_manager_value("FRED_API_KEY")
        return self._secrets["FRED_API_KEY"]

    @property
    def VERTEX_LOCATION(self) -> str:
        return "asia-southeast1"

    @property
    def VERTEX_MODEL(self) -> str:
        return "gemini-3-flash-preview"

settings = Settings()
