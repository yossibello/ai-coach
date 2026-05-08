# AI Coach — Intelligent Cycling Coaching Platform

A full-stack AI coaching platform for cyclists, powered by a **temporal transformer** that learns from your training history to recommend the perfect next workout and predict fitness progression.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│  Next.js 14 Frontend  (TypeScript · Tailwind · Recharts)        │
│  Landing · Auth · Dashboard · AI Coach · Upload · Profile       │
└────────────────────┬────────────────────────────────────────────┘
                     │ REST + JWT
┌────────────────────▼────────────────────────────────────────────┐
│  FastAPI Backend  (Python 3.11)                                  │
│  Auth · Activities · Strava OAuth · Fitness PMC · Coach API     │
└──────┬──────────────────────────┬───────────────────────────────┘
       │                          │
┌──────▼──────┐         ┌─────────▼────────────────────────────┐
│ PostgreSQL  │         │  ML Module (PyTorch)                  │
│ TimescaleDB │         │  ┌──────────────────────────────────┐ │
│ Activities  │         │  │ CyclingTransformer               │ │
│ Fitness PMC │         │  │  • Temporal attention (90 rides) │ │
│ Users       │         │  │  • 6 encoder layers, 8 heads     │ │
└─────────────┘         │  │  • ~50 features per activity     │ │
                        │  │  • Cold-start periodization       │ │
┌─────────────┐         │  └──────────────────────────────────┘ │
│   Redis     │         └───────────────────────────────────────┘
│ Task queue  │
└─────────────┘
```

---

## Features

| Feature | Description |
|---|---|
| **Strava OAuth** | 1-click import of full ride history |
| **GPX / FIT / TCX upload** | Drag-and-drop file parsing |
| **PMC (Performance Management Chart)** | CTL, ATL, TSB with Recharts |
| **Power zones** | Coggan 7-zone classification |
| **Workout classification** | Auto-detect recovery/endurance/tempo/threshold/VO2max |
| **AI Recommendation** | 7-day plan with interval structures and rationale |
| **FTP Forecast** | 4-week FTP delta prediction with confidence interval |
| **Overtraining risk** | Real-time risk score from transformer |
| **Cold-start** | Scientific periodization rules for new users |
| **Event planning** | Periodized peaking for specific goal events |

---

## Getting Started

### Prerequisites
- Docker & Docker Compose
- Node.js 20+ (for local frontend dev)
- Python 3.11+ (for local backend dev)

### 1. Clone and configure

```bash
git clone <repo>
cd ai-coach
cp .env.example .env
# Edit .env — set STRAVA_CLIENT_ID, STRAVA_CLIENT_SECRET, SECRET_KEY
```

### 2. Start with Docker Compose

```bash
docker-compose up -d
```

| Service   | URL                         |
|-----------|-----------------------------|
| Frontend  | http://localhost:3000        |
| Backend   | http://localhost:8000        |
| API Docs  | http://localhost:8000/docs   |
| Postgres  | localhost:5432               |

### 3. Local development (no Docker)

**Backend:**
```bash
cd backend
python -m venv .venv && source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
cp ../.env.example .env
uvicorn app.main:app --reload
```

**Frontend:**
```bash
cd frontend
npm install
cp ../.env.example .env.local
npm run dev
```

---

## Strava Setup

1. Create an app at https://www.strava.com/settings/api
2. Set **Authorization Callback Domain** to `localhost`
3. Copy `Client ID` and `Client Secret` to `.env`
4. Set `STRAVA_REDIRECT_URI=http://localhost:3000/api/strava/callback`

---

## ML Model

### Architecture: CyclingTransformer

```
Input sequence (up to 90 activities):
  [activity_features(46) + profile_features(10)] × T
           ↓
  Linear projection → 128-dim tokens
           ↓
  Time-aware positional encoding (actual days elapsed)
           ↓
  6× TransformerEncoderLayer (Pre-LN, GELU, 8 heads)
           ↓
  Last token representation
           ↓
  ┌──────────────────────────────────┐
  │ Workout type head   (10 classes)  │
  │ Intensity head      (IF target)   │
  │ Duration head       (hours)       │
  │ FTP delta head      (watts)       │
  │ CTL peak head       (load target) │
  │ Risk head           (3 scores)    │
  └──────────────────────────────────┘
```

### Cold Start (< 50 activities)

Uses scientifically grounded periodization rules:
- **Base phase** (>20 weeks out): Zone 2 + Sweet Spot
- **Build phase** (8–20 weeks): Tempo + Threshold
- **Peak phase** (3–8 weeks): Threshold + VO2max + Taper
- **TSB-based overrides**: rest days when form < -30

### Training the model

```bash
# 1. Export your database activities to parquet
python -m ml.data.export_to_parquet \
  --db postgresql://aicoach:secret@localhost/aicoach \
  --output ./ml/data/activities.parquet

# 2. Train
python -m ml.training.train \
  --data ./ml/data/activities.parquet \
  --output ./backend/models/cycling_coach.pt \
  --epochs 100

# The backend automatically loads the model on startup
```

---

## Activity Feature Set (~50 per ride)

| Category | Features |
|---|---|
| Training Load | TSS, CTL, ATL, TSB |
| Distance/Elevation | Duration, km, elevation |
| Power | avg/max/NP (% FTP), IF, VI |
| Heart Rate | avg/max (% HRmax), HR drift, aerobic efficiency |
| Cadence | avg RPM |
| Time in Zones | Z1–Z7 % |
| Environment | temperature, humidity, wind |
| Temporal | day-of-week (sin/cos), month (sin/cos), days since last |
| Workout type | one-hot 10 classes |
| Perceived effort | RPE 1–10 |

Athlete profile features (static): age, weight, height, sex, FTP, max HR, resting HR, experience, goal type, days to event.

---

## Project Structure

```
ai-coach/
├── frontend/                     # Next.js 14 app
│   └── src/
│       ├── app/                  # Pages (App Router)
│       │   ├── page.tsx          # Landing
│       │   ├── (auth)/           # Login / Signup
│       │   └── (app)/            # Dashboard, Coach, Upload, Profile
│       ├── components/           # React components
│       ├── lib/                  # API client, utils
│       └── types/                # TypeScript interfaces
├── backend/                      # FastAPI
│   └── app/
│       ├── api/endpoints/        # Auth, Activities, Strava, Coach, Fitness
│       ├── core/                 # Config, DB, Security, Deps
│       ├── models/               # SQLAlchemy ORM models
│       ├── services/             # File parser, Strava, Metrics (PMC)
│       └── ml/                   # Model, Features, Inference, Cold-start
├── ml/                           # Standalone training pipeline
│   ├── training/                 # Dataset, Train script
│   └── data/                     # Export script
├── docker-compose.yml
└── .env.example
```

---

## Roadmap

- [ ] HRV integration (Garmin, Polar, Whoop)
- [ ] Blood test / biomarker ingestion
- [ ] Garmin Connect OAuth sync
- [ ] Training plan PDF export
- [ ] Power curve (MMP) analysis
- [ ] Race/event calendar integration
- [ ] Multi-sport (triathlon) support
- [ ] Mobile app (React Native)
- [ ] Federated training (privacy-preserving multi-athlete learning)
