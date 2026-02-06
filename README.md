# 🚀 GCP Slack Trading Agent

> A production-grade, serverless stock analysis agent deployed on Google Cloud Run with Slack integration.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![Cloud Run](https://img.shields.io/badge/GCP-Cloud%20Run-4285F4.svg)](https://cloud.google.com/run)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📖 Table of Contents

- [Architecture Overview](#-architecture-overview)
- [Design Patterns Used](#-design-patterns-used)
- [Technical Indicators](#-technical-indicators)
- [ML Prediction Models](#-ml-prediction-models)
- [Orchestration Flow](#-orchestration-flow)
- [Libraries & Dependencies](#-libraries--dependencies)
- [Local Development](#-local-development)
- [Production Deployment (GCP)](#-production-deployment-gcp)
- [GitHub Actions CI/CD](#-github-actions-cicd)

---

## 🏗️ Architecture Overview

```mermaid
flowchart TB
    subgraph Slack["☁️ Slack Workspace"]
        User([👤 User])
        SlashCmd["/analyze AAPL|30"]
    end

    subgraph GCP["🌐 Google Cloud Platform"]
        subgraph CloudRun["Cloud Run (asia-southeast1)"]
            FastAPI["🐍 FastAPI App"]
            BgTask["Background Task"]
        end
        
        subgraph Services["GCP Services"]
            SecretMgr["🔐 Secret Manager"]
            VertexAI["🤖 Vertex AI"]
        end
    end

    subgraph External["📡 External APIs"]
        Alpaca["📈 Alpaca Markets API"]
    end

    User --> SlashCmd
    SlashCmd -->|HTTP POST| FastAPI
    FastAPI -->|Immediate ACK| SlashCmd
    FastAPI -->|Spawn| BgTask
    BgTask -->|Load Secrets| SecretMgr
    BgTask -->|Fetch OHLCV & News| Alpaca
    BgTask -->|LLM Summarization| VertexAI
    BgTask -->|response_url| Slack

    style CloudRun fill:#e3f2fd
    style Services fill:#fff3e0
    style External fill:#e8f5e9
```

---

## 🎨 Design Patterns Used

The codebase employs several **Gang of Four (GoF) design patterns** to ensure maintainability, testability, and separation of concerns.

### 1. **Façade Pattern** — `StockOrchestrator`

> **Purpose**: Provides a unified interface to a set of complex subsystems.

The `StockOrchestrator` class acts as a **Façade** that hides the complexity of:
- `ResearchAgent` (data gathering)
- `PredictionEnsemble` (ML predictions)
- `Alerter` (trend reversal detection)

```python
# orchestrator.py
class StockOrchestrator:
    def __init__(self):
        self.research_agent = ResearchAgent()
        self.ensemble = PredictionEnsemble()
        self.alerter = Alerter()

    async def run_analysis(self, symbol: str, forecast_days: int):
        research = await self.research_agent.gather_full_research(symbol)
        predictions = self.ensemble.get_ensemble_forecast(...)
        reversal = self.alerter.detect_trend_reversal(...)
        return {...}
```

---

### 2. **Strategy Pattern** — `PredictionEnsemble`

> **Purpose**: Defines a family of algorithms and makes them interchangeable.

Each prediction model (`LSTMModel`, `XGBoostModel`, `MonteCarloSimulator`) implements the same interface (`train_and_predict`), allowing the ensemble to swap or combine strategies dynamically.

```python
# models.py
class PredictionEnsemble:
    def __init__(self):
        self.lstm = LSTMModel()        # Strategy 1
        self.xgb = XGBoostModel()      # Strategy 2
        self.mc = MonteCarloSimulator() # Strategy 3

    def get_ensemble_forecast(self, df, forecast_days):
        lstm_pred = self.lstm.train_and_predict(df, forecast_days)
        xgb_pred = self.xgb.train_and_predict(df, forecast_days)
        mc_pred = self.mc.simulate(df, forecast_days)
        # Weighted ensemble: 40% LSTM, 40% XGBoost, 20% Monte Carlo
        return (lstm_pred * 0.4) + (xgb_pred * 0.4) + (mc_pred * 0.2)
```

---

### 3. **Adapter Pattern** — `AlpacaClient`

> **Purpose**: Converts the interface of a class into another interface clients expect.

The `AlpacaClient` wraps the Alpaca-py SDK, adapting its various response formats into a consistent DataFrame interface.

```python
# data_client.py
class AlpacaClient:
    def get_historical_ohlcv(self, symbol: str, days: int = 1260) -> pd.DataFrame:
        # Adapts Alpaca's BarsResponse → pandas DataFrame
        bars = self.data_client.get_stock_bars(request_params)
        return bars.df.xs(symbol)
```

---

### 4. **Singleton Pattern** — `Settings`

> **Purpose**: Ensures a class has only one instance.

The `settings` object is instantiated once at module load and reused across the application to provide centralized configuration.

```python
# config.py
settings = Settings()
settings.load_secrets()  # Loads from GCP Secret Manager
```

---

### 5. **Template Method** — `TechnicalAnalyzer`

> **Purpose**: Defines the skeleton of an algorithm, deferring some steps to subclasses.

Static methods in `TechnicalAnalyzer` follow a template for calculating indicators (fetch data → compute → return), with calculation logic varying per indicator.

---

## 📊 Technical Indicators

All technical indicators are calculated in [`researcher.py`](./app/researcher.py) using **pandas** vectorized operations.

| Indicator | Formula | Period | Usage |
|-----------|---------|--------|-------|
| **SMA 20** | `close.rolling(20).mean()` | 20 days | Short-term trend |
| **EMA 20** | `close.ewm(span=20).mean()` | 20 days | Smoothed trend |
| **RSI** | `100 - (100 / (1 + RS))` where RS = gain/loss | 14 days | Overbought/Oversold |
| **MACD** | `EMA(12) - EMA(26)` | 12/26 days | Momentum |
| **Signal Line** | `MACD.ewm(span=9).mean()` | 9 days | MACD crossover |

### Momentum Regime Classification

The system classifies the stock's **momentum regime** based on its position relative to key moving averages:

```mermaid
flowchart LR
    subgraph Inputs
        SMA50["📈 SMA 50"]
        SMA200["📉 SMA 200"]
        VWAP200W["📊 200-Week VWAP"]
    end

    subgraph Decision["Regime Logic"]
        Check{{"Price > SMA50 AND Price > SMA200?"}}
    end

    subgraph Regimes["Output"]
        Strong["🟢 Strong"]
        Neutral["🟡 Neutral"]
        Weak["🔴 Weak"]
    end

    Inputs --> Check
    Check -->|Both True| Strong
    Check -->|Mixed| Neutral
    Check -->|Both False| Weak
```

#### Streak Statistics

The system tracks how long the current regime has persisted and compares it to historical data:

| Metric | Description |
|--------|-------------|
| `current_streak_days` | Days in current regime |
| `avg_streak_days` | Historical average duration for this regime |
| `max_streak_days` | Longest recorded streak |
| `streak_percentile` | Percentile rank of current streak |

---

## 🤖 ML Prediction Models

The `PredictionEnsemble` combines **three independent models** for robust forecasting:

```mermaid
flowchart TB
    subgraph Data["📥 Input: OHLCV Data"]
        DF[("pd.DataFrame")]
    end

    subgraph Models["🧠 Prediction Models"]
        LSTM["LSTM Neural Network<br/>Weight: 40%"]
        XGB["XGBoost Regressor<br/>Weight: 40%"]
        MC["Monte Carlo Simulation<br/>Weight: 20%"]
    end

    subgraph Output["📤 Output"]
        Ensemble["Weighted Ensemble Forecast"]
        Probs["Probability Percentiles<br/>(p10, p30, p50, p75, p90)"]
    end

    DF --> LSTM
    DF --> XGB
    DF --> MC
    LSTM --> Ensemble
    XGB --> Ensemble
    MC --> Ensemble
    Ensemble --> Probs
```

### Model Details

| Model | Architecture | Training | Purpose |
|-------|--------------|----------|---------|
| **LSTM** | 2 LSTM layers (50 units), Dropout(0.2), Dense | 10 epochs, batch=32 | Captures sequential patterns |
| **XGBoost** | 100 estimators, depth=5, lr=0.1 | Lag features (1-5 days) | Tree-based pattern recognition |
| **Monte Carlo** | 1000 simulations (configurable) | GBM with μ and σ from returns | Probabilistic percentile estimation |

### Monte Carlo Output Structure

The Monte Carlo simulator returns **actual percentiles** from simulation distributions:

| Percentile | Interpretation | Slack Display |
|------------|----------------|---------------|
| **p10** | Bearish extreme — 10% chance price falls below this | 🔴 |
| **p30** | Pessimistic scenario — 30% chance price falls below | 🟠 |
| **p50** | Median expectation — equal chance above/below | 🟡 |
| **p75** | Optimistic scenario — 75% chance price stays below | 🟢 |
| **p90** | Bullish extreme — 90% chance price stays below | 🚀 |

**Output JSON:**
```json
{
  "forecast": [183.2, 184.5, 185.1],
  "final_price": 185.1,
  "percentiles": {
    "p10": 178.50,
    "p30": 181.20,
    "p50": 184.00,
    "p75": 187.30,
    "p90": 191.80
  },
  "interpretation": {
    "bearish_extreme": "10% chance price falls below $178.50",
    "pessimistic": "30% chance price falls below $181.20",
    "median": "50% chance price is around $184.00",
    "optimistic": "75% chance price stays below $187.30",
    "bullish_extreme": "90% chance price stays below $191.80"
  },
  "model_contributions": {
    "lstm": 185.2,
    "xgboost": 184.8,
    "monte_carlo": 185.5
  }
}
```

### Enhanced Weighted Candle Analysis 🆕

The system now features **sophisticated weighted candle analysis** that addresses three critical limitations of simple counting:

#### The Problem with Simple Counting

❌ **Old Method Issues:**
1. **Magnitude blindness**: 4 months of +1% counted same as 2 months of -15% (net bearish ignored!)
2. **Recency blindness**: Equal weight to all periods misses trend reversals
3. **Context blindness**: No awareness of economic events driving volatility

#### The Solution: Weighted Scoring

✅ **New Algorithm:**
```python
weighted_contribution = direction × magnitude × conviction × recency_weight

where:
- direction = +1 (bullish) or -1 (bearish)
- magnitude = abs(change_pct)
- conviction = body_ratio / 100  # Decisive move vs indecision
- recency_weight = 0.95^(lookback - i - 1)  # Latest = 1.0, oldest ≈ 0.57
```

#### Enhanced Metrics

| Metric | Description | Example |
|--------|-------------|----------|
| **Weighted Trend Score** | Sum of all weighted contributions | +35.2 = Bullish |
| **Recent Momentum** | Last 25% of periods (no decay) | -8.3 = Reversal detected! |
| **Volatility Regime** | Current volatility vs historical | "High Volatility" (80th %ile) |
| **Conviction Score** | Average body ratio across periods | 72% = Decisive moves |
| **Economic Events** 🆕 | CPI, unemployment, FOMC tracked via FRED API | 8 months with events |

#### FRED API Integration

Automatic tracking of market-moving economic events:
- 📅 **CPI (Consumer Price Index)** - Monthly inflation data
- 📅 **Unemployment Rate** - Jobs report (first Friday)
- 📅 **FOMC Rate Changes** - Federal Reserve decisions

Months with events are marked with 📅 in the chart.

#### Enhanced Slack Output

```
*📊 Enhanced 12-Month Trend Analysis:*
• Overall Trend: 🟢 *Bullish* (Score: +35.2)
• Recent Momentum: 🔴 *Bearish* (Score: -8.3)  ⚠️ Reversal!
• Volatility: 🔥 High Volatility (4.1%)
• Conviction: 72%
• Economic Events: 8 months tracked 📅

*Monthly Chart:*
```
Jan 2024: 🟢  +5.2% ████████ 📅
Feb 2024: 🟢  +3.1% ██████
Mar 2024: 🔴  -2.1% ████ 📅
Apr 2024: 🟢  +1.8% ███
May 2024: 🔴 -12.3% ████████████ 📅
Jun 2024: 🔴  -8.7% █████████
```
_Simple count: Neutral (3🟢 vs 3🔴)_
_Weighted trend: Bearish (magnitude matters!)_
```

#### Example: Magnitude Weighting in Action

**Scenario:** 4 months of +1% vs 2 months of -15%

| Method | Result | Explanation |
|--------|--------|-------------|
| **Simple Count** | "Bullish" (4 > 2) | ❌ Misleading! |
| **Weighted Score** | -20.4 (Bearish) | ✅ Correctly identifies net bearish trend |

The weighted method properly accounts for the fact that two -15% months **outweigh** four +1% months in actual P&L impact.

---

## 🔄 Orchestration Flow

The orchestration follows a **pipeline pattern** coordinated by Fast API background tasks:

```mermaid
sequenceDiagram
    autonumber
    participant U as 👤 User (Slack)
    participant F as 🐍 FastAPI
    participant R as 📊 ResearchAgent
    participant M as 🤖 PredictionEnsemble
    participant A as 🚨 Alerter
    participant S as ☁️ Slack

    U->>F: /analyze AAPL|30
    F-->>U: ⏳ "Analysis request received!"
    
    activate F
    Note right of F: Background Task Starts
    
    F->>R: gather_full_research("AAPL")
    R->>R: 1. Fetch OHLCV (1260 days)
    R->>R: 2. Calculate Technicals
    R->>R: 3. Calculate Momentum Regime
    R-->>F: research payload
    
    F->>M: get_ensemble_forecast(df, 30)
    M->>M: Train LSTM
    M->>M: Train XGBoost
    M->>M: Run Monte Carlo
    M-->>F: predictions
    
    F->>A: detect_trend_reversal(df)
    A-->>F: reversal_flag
    
    F->>F: format_slack_message()
    F->>S: POST to response_url
    deactivate F
    
    S-->>U: 📈 Full Analysis Report
```

### Error Handling Strategy

Each component has **graceful degradation**:
- If OHLCV fetch fails → Technical analysis shows error
- If momentum fails → Predictions still run with available data
- All errors are captured and status reported to Slack

---

## 📚 Libraries & Dependencies

### Core Framework
| Library | Version | Purpose |
|---------|---------|---------|
| `fastapi` | Latest | Async REST API framework |
| `uvicorn[standard]` | Latest | ASGI server |
| `python-multipart` | Latest | Form data parsing |
| `pydantic-settings` | Latest | Configuration management |

### Data & ML
| Library | Version | Purpose |
|---------|---------|---------|
| `pandas` | Latest | Data manipulation |
| `numpy` | Latest | Numerical computing |
| `scikit-learn` | Latest | ML preprocessing (MinMaxScaler) |
| `xgboost` | Latest | Gradient boosting model |
| `tensorflow` | Latest | LSTM neural network |

### External APIs
| Library | Version | Purpose |
|---------|---------|---------|
| `alpaca-py` | Latest | Market data API |
| `slack-sdk` | Latest | Slack API integration |
| `requests` | Latest | HTTP client |
| `fredapi` | Latest | 🆕 Federal Reserve Economic Data API |

### GCP Integration
| Library | Version | Purpose |
|---------|---------|---------|
| `google-cloud-secret-manager` | Latest | Secret storage |
| `google-cloud-aiplatform` | Latest | Vertex AI access |
| `langchain` | Latest | LLM orchestration |
| `langchain-google-vertexai` | Latest | Vertex AI bindings |

---

## 🛠️ Local Development

### Prerequisites
- Python 3.11+
- Docker (optional, for container testing)
- GCP account with Secret Manager

### Setup

```bash
# 1. Clone and install
git clone <repo-url>
cd gcp-slack-agent-cloud
pip install -r requirements.txt

# 2. Create .env file
cat > .env << EOF
ALPACA_API_KEY=your_key
ALPACA_SECRET_KEY=your_secret
SLACK_BOT_TOKEN=xoxb-...
SLACK_SIGNING_SECRET=...
FRED_API_KEY=your_fred_api_key  # Get free key at https://fred.stlouisfed.org/docs/api/api_key.html
EOF

# 3. Run locally
uvicorn app.main:app --reload --port 8080
```

### Testing

```bash
# Run unit tests
./run_test.sh

# Manual API test
curl -X POST http://localhost:8080/slack/events \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "text=AAPL|30&response_url=http://localhost:8080/test"
```

---

## ☁️ Production Deployment (GCP)

### Architecture Decisions

| Component | Choice | Rationale |
|-----------|--------|-----------|
| **Compute** | Cloud Run | Pay-per-use, auto-scaling, no cluster management |
| **Region** | asia-southeast1 | Low latency for target users |
| **Memory** | 4 GiB | TensorFlow LSTM training requirement |
| **CPU** | 2 vCPUs | Parallel model training |
| **Secrets** | Secret Manager | Secure, auditable, no env vars in code |

### Deployment Commands

```bash
# 1. Initialize GCP project
./init_gcp.sh

# 2. Upload secrets and deploy
./deploy_app.sh

# 3. Verify deployment
./check_resources.sh

# 4. (Optional) Cleanup
./destroy_app.sh
```

### Manual Deployment

```bash
PROJECT_ID=$(gcloud config get-value project)

# Build and push container
gcloud builds submit --tag gcr.io/$PROJECT_ID/slack-trading-agent

# Deploy to Cloud Run
gcloud run deploy slack-trading-agent \
  --image gcr.io/$PROJECT_ID/slack-trading-agent \
  --region asia-southeast1 \
  --memory 4Gi \
  --cpu 2 \
  --timeout 600 \
  --allow-unauthenticated
```

---

## 🔧 GitHub Actions CI/CD

Create `.github/workflows/deploy.yml` for automated deployments:

```yaml
name: Deploy to Cloud Run

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

env:
  PROJECT_ID: ${{ secrets.GCP_PROJECT_ID }}
  SERVICE: slack-trading-agent
  REGION: asia-southeast1

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          
      - name: Install dependencies
        run: pip install -r requirements.txt
        
      - name: Run tests
        run: pytest tests/ -v

  deploy:
    needs: test
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    
    permissions:
      contents: read
      id-token: write

    steps:
      - uses: actions/checkout@v4

      - name: Google Auth
        uses: google-github-actions/auth@v2
        with:
          workload_identity_provider: ${{ secrets.WIF_PROVIDER }}
          service_account: ${{ secrets.WIF_SERVICE_ACCOUNT }}

      - name: Set up Cloud SDK
        uses: google-github-actions/setup-gcloud@v2

      - name: Build and Push Container
        run: |
          gcloud builds submit \
            --tag gcr.io/$PROJECT_ID/$SERVICE \
            --quiet

      - name: Deploy to Cloud Run
        run: |
          gcloud run deploy $SERVICE \
            --image gcr.io/$PROJECT_ID/$SERVICE \
            --region $REGION \
            --memory 4Gi \
            --cpu 2 \
            --timeout 600 \
            --allow-unauthenticated \
            --quiet
```

### Required GitHub Secrets

| Secret | Description |
|--------|-------------|
| `GCP_PROJECT_ID` | Your GCP project ID |
| `WIF_PROVIDER` | Workload Identity Federation provider |
| `WIF_SERVICE_ACCOUNT` | Service account email for deployments |

### Setting Up Workload Identity Federation

```bash
# 1. Create a Workload Identity Pool
gcloud iam workload-identity-pools create github-pool \
  --location global \
  --display-name "GitHub Actions Pool"

# 2. Create a Provider for GitHub
gcloud iam workload-identity-pools providers create-oidc github-provider \
  --location global \
  --workload-identity-pool github-pool \
  --display-name "GitHub Provider" \
  --attribute-mapping "google.subject=assertion.sub,attribute.actor=assertion.actor,attribute.repository=assertion.repository" \
  --issuer-uri "https://token.actions.githubusercontent.com"

# 3. Grant permissions to the service account
gcloud iam service-accounts add-iam-policy-binding $SA_EMAIL \
  --role roles/iam.workloadIdentityUser \
  --member "principalSet://iam.googleapis.com/.../attribute.repository/YOUR_ORG/YOUR_REPO"
```

---

## 📁 Project Structure

```
gcp-slack-agent-cloud/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI entry point
│   ├── orchestrator.py      # Façade - coordinates all subsystems
│   ├── researcher.py        # Technical analysis & data gathering
│   ├── data_client.py       # Alpaca API adapter
│   ├── models.py            # LSTM, XGBoost, Monte Carlo
│   ├── alerter.py           # Trend reversal detection
│   ├── config.py            # GCP Secret Manager integration
│   └── log_config.py        # Logging configuration
├── .env                     # Local environment variables
├── Dockerfile               # Container definition
├── requirements.txt         # Python dependencies
├── deploy_app.sh            # One-click deployment script
├── init_gcp.sh              # GCP project initialization
├── upload_secrets.sh        # Secret Manager sync
├── check_resources.sh       # Resource verification
├── destroy_app.sh           # Cleanup script
└── README.md                # This file
```

---

## 📜 License

MIT License - See [LICENSE](LICENSE) for details.

---

<p align="center">
  <b>Built with ❤️ for quantitative trading enthusiasts</b><br>
  <i>GCP Stock Agent v3.0</i> - Now with Enhanced Weighted Candle Analysis
</p>
