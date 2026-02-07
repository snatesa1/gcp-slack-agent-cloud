from datetime import datetime
from typing import Dict
import os

class StockOrchestrator:
    def __init__(self):
        from .researcher import ResearchAgent, TechnicalAnalyzer
        from .models import PredictionEnsemble
        from .alerter import Alerter
        
        self.ResearchAgent = ResearchAgent
        self.TechnicalAnalyzer = TechnicalAnalyzer
        
        self.research_agent = ResearchAgent()
        self.ensemble = PredictionEnsemble()
        self.alerter = Alerter()

    async def run_analysis(self, symbol: str, forecast_days: int = 30):
        from .config import settings
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
        
        # 5. Enhanced Weighted Candlestick Analysis
        monthly_candles = None
        candles_status = "❌ No data"
        try:
            if df is not None and not df.empty:
                fred_key = settings.FRED_API_KEY
                monthly_candles = self.TechnicalAnalyzer.calculate_weighted_candles(
                    df, 
                    lookback=12, 
                    recency_decay=0.95,
                    fred_api_key=fred_key
                )
                candles_status = "✅ Success"
        except Exception as e:
            print(f"⚠️ Monthly candles failed for {symbol}: {e}")
            candles_status = "❌ Failed"
        
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
            "monthly_candles": monthly_candles,
            "trend_reversal": reversal,
            "timestamp": datetime.now().isoformat(),
            "pred_status": pred_status,
            "candles_status": candles_status
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
            pcts = result['predictions'].get('percentiles', {})
            msg += "*📊 Price Probability Distribution:*\n"
            msg += f"• 🔴 10% (Bearish): ${pcts.get('p10', 0):.2f}\n"
            msg += f"• 🟠 30% (Pessimistic): ${pcts.get('p30', 0):.2f}\n"
            msg += f"• 🟡 50% (Median): ${pcts.get('p50', 0):.2f}\n"
            msg += f"• 🟢 75% (Optimistic): ${pcts.get('p75', 0):.2f}\n"
            msg += f"• 🚀 90% (Bullish): ${pcts.get('p90', 0):.2f}\n"
        
        # Enhanced Monthly Candlestick Analysis
        candles_data = result.get('monthly_candles')
        if candles_data and candles_data.get('candles'):
            summary = candles_data.get('summary', {})
            
            msg += "\n*📊 Enhanced 12-Month Trend Analysis:*\n"
            
            # Weighted metrics first
            weighted_score = summary.get('weighted_trend_score', 0)
            weighted_trend = summary.get('weighted_trend', 'Neutral')
            recent_momentum = summary.get('recent_momentum', 0)
            recent_trend = summary.get('recent_trend', 'Neutral')
            
            trend_emoji = {'Bullish': '🟢', 'Bearish': '🔴', 'Neutral': '🟡'}.get(weighted_trend, '🟡')
            msg += f"• Overall Trend: {trend_emoji} *{weighted_trend}* (Score: {weighted_score:+.1f})\n"
            
            recent_emoji = {'Bullish': '🟢', 'Bearish': '🔴', 'Neutral': '🟡'}.get(recent_trend, '🟡')
            msg += f"• Recent Momentum: {recent_emoji} *{recent_trend}* (Score: {recent_momentum:+.1f})\n"
            
            # Volatility regime
            vol_regime = summary.get('volatility_regime', 'Normal')
            vol_emoji = {'High Volatility': '🔥', 'Low Volatility': '😴', 'Normal': '📊'}.get(vol_regime, '📊')
            msg += f"• Volatility: {vol_emoji} {vol_regime} ({summary.get('current_volatility', 0):.1f}%)\n"
            msg += f"• Conviction: {summary.get('avg_conviction', 0):.0f}%\n"
            
            # Event markers
            if summary.get('has_events'):
                msg += f"• Economic Events: {summary.get('event_months', 0)} months tracked 📅\n"
            
            msg += "\n*Monthly Chart:*\n```\n"
            
            # Show last 6 months for mobile readability
            candles = candles_data['candles'][-6:]
            for candle in candles:
                month = candle['month']
                change = candle['change_pct']
                is_bull = candle['is_bullish']
                events = candle.get('events', [])
                
                # Create ASCII bar
                bar_len = min(int(abs(change) * 2), 10)
                if is_bull:
                    bar = '█' * bar_len
                    emoji = '🟢'
                    sign = '+'
                else:
                    bar = '░' * bar_len
                    emoji = '🔴'
                    sign = ''
                
                event_marker = ' 📅' if events else ''
                msg += f"{month}: {emoji} {sign}{change:>5.1f}% {bar}{event_marker}\n"
            
            msg += "```\n"
            
            # Comparison: simple vs weighted
            simple_trend = summary.get('simple_trend', 'Neutral')
            bull = summary.get('bullish_months', 0)
            bear = summary.get('bearish_months', 0)
            msg += f"_Simple count: {simple_trend} ({bull}🟢 vs {bear}🔴)_\n"
        
        if result['trend_reversal']:
            msg += "\n🚨 *Alert: Technical Trend Reversal Detected!*"
            
        msg += "\n*System Status:*\n"
        msg += f"• Techs: {status.get('technicals', '❓')}\n"
        msg += f"• Momentum: {status.get('momentum', '❓')}\n"
        msg += f"• Candles: {result.get('candles_status', '❓')}\n"
        msg += f"• Preds: {result.get('pred_status', '❓')}\n"
            
        msg += "\n_Automated GCP Stock Agent v2.5_"
        return msg

