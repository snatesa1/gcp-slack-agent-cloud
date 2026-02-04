from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
import pandas as pd
from .config import settings

class Alerter:
    def __init__(self):
        self.client = WebClient(token=settings.SLACK_BOT_TOKEN)

    def detect_trend_reversal(self, df: pd.DataFrame) -> bool:
        if len(df) < 25:
            return False
            
        sma_5 = df['close'].rolling(window=5).mean()
        sma_20 = df['close'].rolling(window=20).mean()
        
        # Bullish or Bearish Crossover
        crossover = (sma_5.iloc[-2] <= sma_20.iloc[-2] and sma_5.iloc[-1] > sma_20.iloc[-1]) or \
                    (sma_5.iloc[-2] >= sma_20.iloc[-2] and sma_5.iloc[-1] < sma_20.iloc[-1])
                    
        return crossover

    async def send_slack_notification(self, channel_id: str, message: str):
        try:
            # await self.client.chat_postMessage(...) # WebClient is synchronous mostly, but can be used in async
            response = self.client.chat_postMessage(
                channel=channel_id,
                text=message
            )
            return response
        except SlackApiError as e:
            print(f"Slack notification error: {e}")
            return None
