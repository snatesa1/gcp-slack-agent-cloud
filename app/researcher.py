import pandas as pd
import numpy as np
from typing import List, Dict
from .data_client import AlpacaClient

class TechnicalAnalyzer:
    @staticmethod
    def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
        df['SMA_20'] = df['close'].rolling(window=20).mean()
        df['EMA_20'] = df['close'].ewm(span=20, adjust=False).mean()
        
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        return df
    
    @staticmethod
    def calculate_momentum_regime(df: pd.DataFrame) -> Dict:
        """
        Calculates momentum regime and historical persistence statistics.
        Returns current regime, streak, and historical context.
        """
        # Calculate moving averages
        sma50 = df['close'].rolling(50).mean()
        sma200 = df['close'].rolling(200).mean()
        
        # Calculate 200-week VWAP (approx 1000 trading days)
        if 'volume' in df.columns:
            vwap200w = (df['close'] * df['volume']).rolling(1000, min_periods=200).sum() / \
                       df['volume'].rolling(1000, min_periods=200).sum()
        else:
            vwap200w = df['close'].rolling(1000, min_periods=200).mean()
        
        # Determine position relative to MAs
        above_50 = df['close'] > sma50
        above_200 = df['close'] > sma200
        above_vwap = df['close'] > vwap200w
        
        # Regime classification
        def get_regime(a50: bool, a200: bool) -> str:
            if a50 and a200:
                return "Strong"
            elif not a50 and not a200:
                return "Weak"
            else:
                return "Neutral"
        
        # Current state
        current_above_50 = above_50.iloc[-1] if not above_50.empty else False
        current_above_200 = above_200.iloc[-1] if not above_200.empty else False
        current_above_vwap = above_vwap.iloc[-1] if not above_vwap.empty else False
        current_regime = get_regime(current_above_50, current_above_200)
        
        # Calculate streak statistics for each regime type
        regimes = pd.Series([get_regime(a50, a200) for a50, a200 in zip(above_50, above_200)], index=df.index)
        regimes = regimes.dropna()
        
        streak_stats = TechnicalAnalyzer._calculate_streak_stats(regimes, current_regime)
        
        # Get latest values
        latest = df.iloc[-1]
        
        return {
            "regime": current_regime,
            "above_50_ma": bool(current_above_50),
            "above_200_ma": bool(current_above_200),
            "above_200w_vwap": bool(current_above_vwap),
            "current_streak_days": streak_stats["current_streak"],
            "avg_streak_days": streak_stats["avg_streak"],
            "max_streak_days": streak_stats["max_streak"],
            "streak_percentile": streak_stats["percentile"],
            "ma_values": {
                "sma_50": round(sma50.iloc[-1], 2) if not sma50.empty else None,
                "sma_200": round(sma200.iloc[-1], 2) if not sma200.empty else None,
                "vwap_200w": round(vwap200w.iloc[-1], 2) if not vwap200w.empty and not np.isnan(vwap200w.iloc[-1]) else None
            }
        }
    
    @staticmethod
    def _calculate_streak_stats(regimes: pd.Series, current_regime: str) -> Dict:
        """
        Calculates current streak length and historical statistics for similar regimes.
        """
        if regimes.empty:
            return {"current_streak": 0, "avg_streak": 0, "max_streak": 0, "percentile": 0}
        
        # Find all streaks of the current regime type
        regime_mask = regimes == current_regime
        
        # Calculate streak lengths using group-by on consecutive values
        streak_groups = (regime_mask != regime_mask.shift()).cumsum()
        streak_lengths = regime_mask.groupby(streak_groups).cumsum()
        
        # Get streaks only for the matching regime
        matching_streaks = streak_lengths[regime_mask]
        
        # Current streak is the last value
        current_streak = int(matching_streaks.iloc[-1]) if not matching_streaks.empty else 0
        
        # Get completed streak lengths (when regime changes)
        regime_changes = (regimes != regimes.shift(-1)) & (regimes == current_regime)
        completed_streaks = streak_lengths[regime_changes].values
        
        if len(completed_streaks) == 0:
            return {
                "current_streak": current_streak,
                "avg_streak": current_streak,
                "max_streak": current_streak,
                "percentile": 50
            }
        
        avg_streak = int(np.mean(completed_streaks))
        max_streak = int(np.max(completed_streaks))
        
        # Percentile of current streak among historical
        percentile = int(np.percentile(completed_streaks, 100 * (current_streak < completed_streaks).mean()))
        percentile = min(100, int(100 * (np.sum(completed_streaks <= current_streak) / len(completed_streaks))))
        
        return {
            "current_streak": current_streak,
            "avg_streak": avg_streak,
            "max_streak": max_streak,
            "percentile": percentile
        }


class ResearchAgent:
    def __init__(self):
        self.alpaca = AlpacaClient()
        self.tech_analyzer = TechnicalAnalyzer()

    async def gather_full_research(self, symbol: str):
        ohlcv = None
        
        # 1. Technical Analysis
        try:
            ohlcv = self.alpaca.get_historical_ohlcv(symbol)
            ohlcv_with_indicators = self.tech_analyzer.calculate_indicators(ohlcv.copy())
            technicals = ohlcv_with_indicators.tail(1).to_dict('records')[0]
            tech_status = "✅ Success"
        except Exception as e:
            print(f"⚠️ Technical analysis failed for {symbol}: {e}")
            technicals = {"error": str(e)}
            tech_status = "❌ Failed"
        
        # 2. Momentum Analysis
        momentum = None
        momentum_status = "❌ No data"
        if ohlcv is not None:
            try:
                momentum = self.tech_analyzer.calculate_momentum_regime(ohlcv)
                momentum_status = "✅ Success"
            except Exception as e:
                print(f"⚠️ Momentum analysis failed for {symbol}: {e}")
                momentum = {"error": str(e)}
                momentum_status = "❌ Failed"
        
        # 3. Fundamentals
        try:
            fundamentals = self.alpaca.get_fundamentals(symbol)
            fund_status = "✅ Success"
        except Exception as e:
            print(f"⚠️ Fundamentals fetching failed for {symbol}: {e}")
            fundamentals = {"error": str(e)}
            fund_status = "❌ Failed"
        
        return {
            "symbol": symbol,
            "technicals": technicals,
            "momentum": momentum,
            "fundamentals": fundamentals,
            "ohlcv": ohlcv,  # Pass for prediction use
            "status": {
                "technicals": tech_status,
                "momentum": momentum_status,
                "fundamentals": fund_status
            }
        }
