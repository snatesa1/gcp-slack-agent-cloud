from .researcher import ResearchAgent
from .models import PredictionEnsemble
from .alerter import Alerter
from datetime import datetime
from typing import Dict

class StockOrchestrator:
    def __init__(self):
        self.research_agent = ResearchAgent()
        self.ensemble = PredictionEnsemble()
        self.alerter = Alerter()

    async def run_analysis(self, symbol: str, forecast_days: int = 30):
        # 1. Gather Research (Technicals, Momentum & Fundamentals)
        research = await self.research_agent.gather_full_research(symbol)
        
        # 2. Run Predictions (Reuse OHLCV from research)
        predictions = None
        last_price = 0.0
        df = research.get('ohlcv')
        
        try:
            if df is not None and not df.empty:
                last_price = df['close'].iloc[-1]
                predictions = self.ensemble.get_ensemble_forecast(df, forecast_days)
                pred_status = "✅ Success"
            else:
                pred_status = "❌ No OHLCV data"
        except Exception as e:
            print(f"⚠️ Predictions failed for {symbol}: {e}")
            pred_status = f"❌ Failed: {str(e)}"
        
        # 3. Formulate Recommendation
        rating = "Unknown"
        expected_return = "N/A"
        if predictions:
            try:
                target_price = predictions['forecast'][-1]
                ret = (target_price - last_price) / last_price
                expected_return = f"{ret:.2%}"
                
                rating = "Hold"
                if ret > 0.10:
                    rating = "Must-Buy 🚀"
                elif ret > 0.05:
                    rating = "Buy 📈"
            except:
                pass
        
        # 4. Check for alerts
        reversal = False
        try:
            if df is not None:
                reversal = self.alerter.detect_trend_reversal(df)
        except:
            pass
        
        # Remove ohlcv from research before returning (too large for JSON)
        research_copy = {k: v for k, v in research.items() if k != 'ohlcv'}
        
        return {
            "symbol": symbol,
            "current_price": last_price,
            "rating": rating,
            "expected_return": expected_return,
            "research": research_copy,
            "predictions": predictions,
            "momentum": research.get('momentum'),
            "trend_reversal": reversal,
            "timestamp": datetime.now().isoformat(),
            "pred_status": pred_status
        }

    def format_slack_message(self, result: Dict):
        """Formats for Slack mobile with status indicators."""
        res_info = result['research']
        status = res_info.get('status', {})
        
        msg = f"*Production Analysis: {result['symbol']}*\n"
        if result['current_price'] > 0:
            msg += f"Price: ${result['current_price']:.2f} | Rating: *{result['rating']}*\n"
            msg += f"Exp. Return ({len(result['predictions']['forecast']) if result['predictions'] else '?' }d): {result['expected_return']}\n\n"
        else:
            msg += "❌ Price data unavailable.\n\n"
        
        # Momentum Analysis Section
        momentum = result.get('momentum')
        if momentum and not momentum.get('error'):
            regime = momentum.get('regime', 'Unknown')
            regime_emoji = {'Strong': '🟢', 'Neutral': '🟡', 'Weak': '🔴'}.get(regime, '⚪')
            
            msg += f"*Momentum Analysis:*\n"
            msg += f"• Regime: {regime_emoji} *{regime}*"
            
            # Position indicators
            positions = []
            if momentum.get('above_50_ma'):
                positions.append("↑50MA")
            else:
                positions.append("↓50MA")
            if momentum.get('above_200_ma'):
                positions.append("↑200MA")
            else:
                positions.append("↓200MA")
            if momentum.get('above_200w_vwap'):
                positions.append("↑VWAP")
            
            msg += f" ({', '.join(positions)})\n"
            
            # Streak info
            current_streak = momentum.get('current_streak_days', 0)
            avg_streak = momentum.get('avg_streak_days', 0)
            max_streak = momentum.get('max_streak_days', 0)
            percentile = momentum.get('streak_percentile', 0)
            
            msg += f"• Current Streak: *{current_streak}* days"
            if percentile >= 80:
                msg += " 🔥"  # Extended streak
            elif percentile <= 20:
                msg += " 🆕"  # New regime
            msg += f"\n"
            msg += f"• Avg Duration: {avg_streak}d | Max: {max_streak}d\n\n"
        
        if result['predictions']:
            msg += "*Forecast Range (50% Prob):*\n"
            msg += f"${result['predictions']['probs']['50%_range'][0]:.2f} - ${result['predictions']['probs']['50%_range'][1]:.2f}\n"
        
        if result['trend_reversal']:
            msg += "\n🚨 *Alert: Technical Trend Reversal Detected!*"
            
        msg += "\n*System Status:*\n"
        msg += f"• Techs: {status.get('technicals', '❓')}\n"
        msg += f"• Momentum: {status.get('momentum', '❓')}\n"
        msg += f"• Preds: {result.get('pred_status', '❓')}\n"
            
        msg += "\n_Automated GCP Stock Agent v2.3_"
        return msg

