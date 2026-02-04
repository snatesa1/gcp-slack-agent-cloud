from fastapi import FastAPI, Request, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import logging
import os
import requests
from . import log_config
from .orchestrator import StockOrchestrator
from .alerter import Alerter

app = FastAPI(title="GCP Production Stock Agent")
orchestrator = StockOrchestrator()
alerter = Alerter()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.get("/health")
async def health():
    return {"status": "ok", "environment": "production" if os.getenv("K_SERVICE") else "development"}

async def run_analysis_task(text: str, response_url: str):
    """
    Background task that handles parsing, analysis, and Slack delivery.
    """
    symbol = "UNKNOWN"
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
            requests.post(response_url, json={"text": "❌ Usage: `/analyze TICKER|DAYS`"}, timeout=10)
            return

        logger.info(f"🚀 Background task started for {symbol}")
        
        # 2. Run the heavy analysis
        # The orchestrator now has its own fallbacks, so this should rarely throw
        result = await orchestrator.run_analysis(symbol, days)
        message = orchestrator.format_slack_message(result)
        
        # 3. Deliver final result
        resp = requests.post(response_url, json={
            "text": message, 
            "replace_original": "false",
            "response_type": "in_channel"
        }, timeout=10)
        resp.raise_for_status()
        logger.info(f"✅ Analysis for {symbol} delivered successfully.")

    except Exception as e:
        logger.error(f"❌ Critical error in background task for {symbol}: {e}")
        try:
            requests.post(response_url, json={
                "text": f"❌ *Analysis Failed for {symbol}*\nReason: `{str(e)}`",
                "replace_original": "false"
            }, timeout=10)
        except Exception as inner_e:
            logger.error(f"🔥 Failed to send error notification to Slack: {inner_e}")

@app.post("/slack/events")
async def slack_events(background_tasks: BackgroundTasks, request: Request):
    """
    Immediate response endpoint for Slack Slash Commands.
    """
    try:
        # Fast extraction of response_url and text
        body = await request.form()
        text = body.get("text", "")
        response_url = body.get("response_url")
        
        if not response_url:
            return {"text": "Missing response_url"}

        # Offload EVERYTHING to background
        background_tasks.add_task(run_analysis_task, text, response_url)
        
        # Immediate 200 OK within 3 seconds
        return JSONResponse(content={
            "response_type": "ephemeral",
            "text": "🔍 Analysis request received! I'm crunching the numbers and news... hold tight! ⏳"
        })
    except Exception as e:
        logger.error(f"Event error: {e}")
        return {"text": "Internal error."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
