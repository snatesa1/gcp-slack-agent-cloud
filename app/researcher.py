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
    
    @staticmethod
    def calculate_monthly_candles(df: pd.DataFrame, months: int = 6) -> Dict:
        """
        LEGACY METHOD - Use calculate_weighted_candles for enhanced analysis.
        Resamples daily OHLCV data to monthly candlesticks.
        Returns last N months with bullish/bearish classification.
        
        Args:
            df: DataFrame with OHLCV columns
            months: Number of months to return (default: 6)
            
        Returns:
            Dict with monthly candles and summary statistics
        """
        if df is None or df.empty:
            return {"error": "No data available", "candles": []}
        
        try:
            # Ensure index is datetime
            df_copy = df.copy()
            if not isinstance(df_copy.index, pd.DatetimeIndex):
                df_copy.index = pd.to_datetime(df_copy.index)
            
            # Resample to monthly OHLC
            monthly = df_copy.resample('ME').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum' if 'volume' in df_copy.columns else 'first'
            }).dropna()
            
            # Get last N months
            monthly = monthly.tail(months)
            
            candles = []
            for idx, row in monthly.iterrows():
                open_price = row['open']
                close_price = row['close']
                high_price = row['high']
                low_price = row['low']
                
                # Calculate body and wick sizes
                body_pct = ((close_price - open_price) / open_price) * 100
                range_pct = ((high_price - low_price) / low_price) * 100
                
                # Bullish = close > open, Bearish = close < open
                is_bullish = close_price >= open_price
                
                # Strength based on body size relative to range
                body_size = abs(close_price - open_price)
                total_range = high_price - low_price
                body_ratio = (body_size / total_range * 100) if total_range > 0 else 0
                
                candles.append({
                    "month": idx.strftime("%b %Y"),
                    "open": round(open_price, 2),
                    "high": round(high_price, 2),
                    "low": round(low_price, 2),
                    "close": round(close_price, 2),
                    "is_bullish": is_bullish,
                    "change_pct": round(body_pct, 2),
                    "body_ratio": round(body_ratio, 1),
                    "range_pct": round(range_pct, 2)
                })
            
            # Summary stats
            bullish_count = sum(1 for c in candles if c['is_bullish'])
            bearish_count = len(candles) - bullish_count
            avg_gain = np.mean([c['change_pct'] for c in candles if c['is_bullish']]) if bullish_count > 0 else 0
            avg_loss = np.mean([c['change_pct'] for c in candles if not c['is_bullish']]) if bearish_count > 0 else 0
            
            return {
                "candles": candles,
                "summary": {
                    "bullish_months": bullish_count,
                    "bearish_months": bearish_count,
                    "avg_bullish_gain": round(avg_gain, 2),
                    "avg_bearish_loss": round(avg_loss, 2),
                    "trend": "Bullish" if bullish_count > bearish_count else "Bearish" if bearish_count > bullish_count else "Neutral"
                }
            }
        except Exception as e:
            return {"error": str(e), "candles": []}
    
    @staticmethod
    def calculate_weighted_candles(
        df: pd.DataFrame, 
        lookback: int = 12,
        recency_decay: float = 0.95,
        fred_api_key: str = None
    ) -> Dict:
        """
        Enhanced candle analysis with magnitude weighting, recency bias, and event markers.
        
        Args:
            df: DataFrame with OHLCV columns
            lookback: Number of months to analyze (default: 12)
            recency_decay: Decay factor for recency weighting (default: 0.95)
            fred_api_key: FRED API key for economic events (optional)
            
        Returns:
            Dict with weighted analysis including:
            - Individual candle data with events
            - Weighted trend score (magnitude * conviction * recency)
            - Recent momentum score
            - Volatility regime classification
        """
        if df is None or df.empty:
            return {"error": "No data available", "candles": []}
        
        try:
            # Ensure index is datetime
            df_copy = df.copy()
            if not isinstance(df_copy.index, pd.DatetimeIndex):
                df_copy.index = pd.to_datetime(df_copy.index)
            
            # Resample to monthly OHLC
            monthly = df_copy.resample('ME').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum' if 'volume' in df_copy.columns else 'first'
            }).dropna()
            
            # Get last N months
            monthly = monthly.tail(lookback)
            
            if monthly.empty:
                return {"error": "Insufficient data", "candles": []}
            
            # Fetch economic events if FRED API key provided
            events = {}
            if fred_api_key:
                events = TechnicalAnalyzer._fetch_economic_events(
                    fred_api_key,
                    monthly.index[0],
                    monthly.index[-1]
                )
            
            candles = []
            volatilities = []
            weighted_score = 0.0
            recent_score = 0.0
            recent_start = lookback - (lookback // 4)  # Last 25% of periods
            
            for i, (idx, row) in enumerate(monthly.iterrows()):
                open_price = row['open']
                close_price = row['close']
                high_price = row['high']
                low_price = row['low']
                
                # Price metrics
                body_pct = ((close_price - open_price) / open_price) * 100
                range_pct = ((high_price - low_price) / low_price) * 100
                is_bullish = close_price >= open_price
                
                # Conviction: body size relative to total range
                body_size = abs(close_price - open_price)
                total_range = high_price - low_price
                body_ratio = (body_size / total_range * 100) if total_range > 0 else 0
                conviction = body_ratio / 100.0  # Normalize to 0-1
                
                # Recency weight: most recent = 1.0, oldest decays exponentially
                recency_weight = recency_decay ** (lookback - i - 1)
                
                # Direction: +1 bullish, -1 bearish
                direction = 1 if is_bullish else -1
                
                # Magnitude: absolute change percentage
                magnitude = abs(body_pct)
                
                # Weighted contribution to trend score
                contribution = direction * magnitude * conviction * recency_weight
                weighted_score += contribution
                
                # Recent momentum (last 25% of periods, no decay)
                if i >= recent_start:
                    recent_score += direction * magnitude * conviction
                
                # Volatility tracking
                volatilities.append(range_pct)
                
                # Check for economic events
                month_key = idx.strftime('%Y-%m')
                event_list = events.get(month_key, [])
                
                candles.append({
                    "month": idx.strftime("%b %Y"),
                    "open": round(open_price, 2),
                    "high": round(high_price, 2),
                    "low": round(low_price, 2),
                    "close": round(close_price, 2),
                    "is_bullish": is_bullish,
                    "change_pct": round(body_pct, 2),
                    "body_ratio": round(body_ratio, 1),
                    "range_pct": round(range_pct, 2),
                    "conviction": round(conviction, 2),
                    "recency_weight": round(recency_weight, 2),
                    "weighted_contribution": round(contribution, 2),
                    "events": event_list
                })
            
            # Summary statistics
            bullish_count = sum(1 for c in candles if c['is_bullish'])
            bearish_count = len(candles) - bullish_count
            avg_gain = np.mean([c['change_pct'] for c in candles if c['is_bullish']]) if bullish_count > 0 else 0
            avg_loss = np.mean([c['change_pct'] for c in candles if not c['is_bullish']]) if bearish_count > 0 else 0
            avg_volatility = np.mean(volatilities)
            
            # Volatility regime classification
            vol_80th = np.percentile(volatilities, 80)
            vol_20th = np.percentile(volatilities, 20)
            latest_vol = volatilities[-1]
            
            if latest_vol > vol_80th:
                vol_regime = "High Volatility"
            elif latest_vol < vol_20th:
                vol_regime = "Low Volatility"
            else:
                vol_regime = "Normal"
            
            # Conviction score: average body ratio
            avg_conviction = np.mean([c['conviction'] for c in candles])
            
            return {
                "candles": candles,
                "summary": {
                    # Raw counts (legacy compatibility)
                    "bullish_months": bullish_count,
                    "bearish_months": bearish_count,
                    "avg_bullish_gain": round(avg_gain, 2),
                    "avg_bearish_loss": round(avg_loss, 2),
                    "simple_trend": "Bullish" if bullish_count > bearish_count else "Bearish" if bearish_count > bullish_count else "Neutral",
                    
                    # Enhanced weighted metrics
                    "weighted_trend_score": round(weighted_score, 2),
                    "weighted_trend": "Bullish" if weighted_score > 0 else "Bearish" if weighted_score < 0 else "Neutral",
                    "recent_momentum": round(recent_score, 2),
                    "recent_trend": "Bullish" if recent_score > 0 else "Bearish" if recent_score < 0 else "Neutral",
                    
                    # Volatility metrics
                    "volatility_regime": vol_regime,
                    "avg_volatility": round(avg_volatility, 2),
                    "current_volatility": round(latest_vol, 2),
                    
                    # Conviction
                    "avg_conviction": round(avg_conviction * 100, 1),  # Convert back to percentage
                    
                    # Event awareness
                    "has_events": len(events) > 0,
                    "event_months": len(events)
                }
            }
        except Exception as e:
            return {"error": str(e), "candles": []}
    
    @staticmethod
    def _fetch_economic_events(api_key: str, start_date, end_date) -> Dict:
        """
        Fetches economic event data from FRED API.
        Returns dict mapping 'YYYY-MM' to list of event descriptions.
        """
        try:
            from fredapi import Fred
            from datetime import datetime
            
            fred = Fred(api_key=api_key)
            events = {}
            
            # CPI (Consumer Price Index) - released monthly around 13th
            try:
                cpi = fred.get_series('CPIAUCSL', start_date, end_date)
                for date, value in cpi.items():
                    month_key = date.strftime('%Y-%m')
                    if month_key not in events:
                        events[month_key] = []
                    events[month_key].append({
                        "type": "CPI",
                        "value": round(value, 2),
                        "impact": "high"
                    })
            except Exception as e:
                print(f"⚠️ Failed to fetch CPI data: {e}")
            
            # Unemployment Rate - released first Friday of each month
            try:
                unemployment = fred.get_series('UNRATE', start_date, end_date)
                for date, value in unemployment.items():
                    month_key = date.strftime('%Y-%m')
                    if month_key not in events:
                        events[month_key] = []
                    events[month_key].append({
                        "type": "Unemployment",
                        "value": round(value, 2),
                        "impact": "high"
                    })
            except Exception as e:
                print(f"⚠️ Failed to fetch unemployment data: {e}")
            
            # Federal Funds Rate (FOMC decisions)
            try:
                fed_funds = fred.get_series('FEDFUNDS', start_date, end_date)
                prev_rate = None
                for date, value in fed_funds.items():
                    if prev_rate is not None and value != prev_rate:
                        month_key = date.strftime('%Y-%m')
                        if month_key not in events:
                            events[month_key] = []
                        events[month_key].append({
                            "type": "FOMC Rate Change",
                            "value": round(value, 2),
                            "change": round(value - prev_rate, 2),
                            "impact": "critical"
                        })
                    prev_rate = value
            except Exception as e:
                print(f"⚠️ Failed to fetch Fed Funds data: {e}")
            
            return events
        except ImportError:
            print("⚠️ fredapi not installed. Install with: pip install fredapi")
            return {}
        except Exception as e:
            print(f"⚠️ Error fetching economic events: {e}")
            return {}



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
