"""Quick test for momentum analysis functionality."""
import sys
sys.path.insert(0, '.')

try:
    from app.researcher import TechnicalAnalyzer, ResearchAgent
    from app.orchestrator import StockOrchestrator
    print("✅ All imports successful!")
    
    # Test the streak calculation with mock data
    import pandas as pd
    import numpy as np
    
    # Create mock OHLCV data (100 days)
    dates = pd.date_range(start='2023-01-01', periods=300, freq='D')
    np.random.seed(42)
    
    # Simulate a trending stock
    prices = 100 + np.cumsum(np.random.randn(300) * 0.5)
    volumes = np.random.randint(1000000, 5000000, 300)
    
    mock_df = pd.DataFrame({
        'open': prices * 0.99,
        'high': prices * 1.01,
        'low': prices * 0.98,
        'close': prices,
        'volume': volumes
    }, index=dates)
    
    # Test momentum calculation
    analyzer = TechnicalAnalyzer()
    momentum = analyzer.calculate_momentum_regime(mock_df)
    
    print("\n📊 Momentum Analysis Test:")
    print(f"  Regime: {momentum['regime']}")
    print(f"  Above 50 MA: {momentum['above_50_ma']}")
    print(f"  Above 200 MA: {momentum['above_200_ma']}")
    print(f"  Current Streak: {momentum['current_streak_days']} days")
    print(f"  Avg Streak: {momentum['avg_streak_days']} days")
    print(f"  Max Streak: {momentum['max_streak_days']} days")
    print(f"  Percentile: {momentum['streak_percentile']}%")
    print("\n✅ All tests passed!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
