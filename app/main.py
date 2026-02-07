import hmac
import hashlib
import time
from fastapi import FastAPI, Request, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import logging
import os
import requests
from . import log_config

app = FastAPI(title="GCP Production Stock Agent")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def verify_slack_signature(request: Request):
    """
    Verifies the signature of the request from Slack.
    Follows: https://api.slack.com/authentication/verifying-requests-from-slack
    """
    from .config import settings
    
    timestamp = request.headers.get("X-Slack-Request-Timestamp")
    signature = request.headers.get("X-Slack-Signature")
    
    if not timestamp or not signature:
        return False
        
    # 1. Check if the request is older than 5 minutes
    if abs(time.time() - int(timestamp)) > 60 * 5:
        return False
        
    # 2. Reconstruct the base string
    body = await request.body()
    sig_basestring = f"v0:{timestamp}:{body.decode('utf-8')}"
    
    # 3. Create hmac-sha256 hash
    my_signature = "v0=" + hmac.new(
        settings.SLACK_SIGNING_SECRET.encode("utf-8"),
        sig_basestring.encode("utf-8"),
        hashlib.sha256
    ).hexdigest()
    
    # 4. Compare signatures using constant-time comparison
    return hmac.compare_digest(my_signature, signature)

@app.get("/health")
async def health():
    return {"status": "ok", "environment": "production" if os.getenv("K_SERVICE") else "development"}

async def send_slack_message(channel: str, text: str):
    """Utility to send a message to a Slack channel using the bot token."""
    from .config import settings
    try:
        resp = requests.post(
            "https://slack.com/api/chat.postMessage",
            headers={"Authorization": f"Bearer {settings.SLACK_BOT_TOKEN}"},
            json={"channel": channel, "text": text},
            timeout=10
        )
        resp.raise_for_status()
    except Exception as e:
        logger.error(f"🔥 Failed to send Slack message: {e}")

async def run_analysis_task(text: str, response_url: str = None, channel_id: str = None):
    """
    Background task that handles parsing, analysis, and Slack delivery.
    """
    from .orchestrator import StockOrchestrator
    from .alerter import Alerter
    
    symbol = "UNKNOWN"
    orchestrator = StockOrchestrator()
    alerter = Alerter()
    
    try:
        # 1. Parse Input
        clean_text = text.strip() if text else ""
        if "|" in clean_text:
            parts = clean_text.split("|")
            symbol = parts[0].strip().upper()
            try:
                days = int(parts[1].replace("days", "").strip())
            except:
                days = 30
        elif clean_text:
            symbol = clean_text.upper()
            days = 30
        else:
            symbol = "UNKNOWN"
            days = 30

        if not symbol or symbol == "UNKNOWN":
            msg = "❌ Usage: `/analyze TICKER|DAYS` or just message me a TICKER."
            if response_url:
                requests.post(response_url, json={"text": msg}, timeout=3)
            elif channel_id:
                await send_slack_message(channel_id, msg)
            return

        logger.info(f"🚀 Background task started for {symbol}")

        # 2. Run the heavy analysis
        result = await orchestrator.run_analysis(symbol, days)
        message = orchestrator.format_slack_message(result)
        
        # 3. Deliver final result
        if response_url:
            resp = requests.post(response_url, json={
                "text": message, 
                "replace_original": "false",
                "response_type": "in_channel"
            }, timeout=10)
            resp.raise_for_status()
        elif channel_id:
            await send_slack_message(channel_id, message)
            
        logger.info(f"✅ Analysis for {symbol} delivered successfully.")

    except Exception as e:
        logger.error(f"❌ Critical error in background task for {symbol}: {e}")
        error_msg = f"❌ *Analysis Failed for {symbol}*\nReason: `{str(e)}`"
        if response_url:
            try:
                requests.post(response_url, json={
                    "text": error_msg,
                    "replace_original": "false"
                }, timeout=10)
            except Exception as inner_e:
                logger.error(f"🔥 Failed to send error notification to Slack: {inner_e}")
        elif channel_id:
            await send_slack_message(channel_id, error_msg)

@app.post("/")
@app.post("/slack/events")
async def slack_events(background_tasks: BackgroundTasks, request: Request):
    """
    Unified endpoint for Slack Slash Commands, Events API (DMs), and Interactivity.
    """
    # 1. Signature Verification (Security Best Practice)
    if not await verify_slack_signature(request):
        logger.warning("🚨 Invalid Slack signature.")
        # We return 401 but some documentation suggests just ignoring it to prevent enumeration.
        # However, for debugging 401 is better.
        raise HTTPException(status_code=401, detail="Invalid signature")

    try:
        # 2. Detect Content Type & Payload Type
        content_type = request.headers.get("content-type", "")
        
        body_json = {}
        is_interactivity = False
        
        if "application/json" in content_type:
            body_json = await request.json()
        else:
            # Potentially Form Data (Slash Command or Interactivity)
            form_data = await request.form()
            if "payload" in form_data:
                import json
                body_json = json.loads(form_data["payload"])
                is_interactivity = True
            else:
                # Standard Slash Command
                text = form_data.get("text", "")
                response_url = form_data.get("response_url")
                if response_url:
                    background_tasks.add_task(run_analysis_task, text, response_url=response_url)
                    return JSONResponse(content={
                        "response_type": "ephemeral",
                        "text": "🔍 Analysis request received! Crunching numbers... ⏳"
                    })

        # 2. Handle URL Verification Challenge
        if body_json.get("type") == "url_verification":
            return {"challenge": body_json.get("challenge")}

        # 3. Handle Events (e.g., Direct Messages)
        if body_json.get("type") == "event_callback":
            event = body_json.get("event", {})
            if event.get("type") == "message" and not event.get("bot_id"):
                text = event.get("text", "")
                channel = event.get("channel")
                # Trigger analysis if text looks like a ticker
                if len(text.strip().split()) == 1:
                    background_tasks.add_task(run_analysis_task, text, channel_id=channel)
                    return {"status": "ok"}

        # 4. Handle Interactivity (e.g., Button Clicks)
        if is_interactivity:
            # For now, just log interactivity. We can add specific button handlers here.
            logger.info(f"Interactivity received: {body_json.get('type')}")
            return {"status": "ok"}

        return {"status": "ignored"}

    except Exception as e:
        logger.error(f"Event parsing error: {e}")
        return {"text": "Internal error."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
