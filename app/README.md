# VN-Rating — Quick Start Guide

## Prerequisites
- Docker Desktop installed and running
- At least 4GB RAM available for Docker

## Run with Docker Compose (Recommended)

```bash
cd e:\thesis\app
docker compose up --build
```

First build may take **5-8 minutes** (installs Python deps + Node deps + compiles Next.js).

- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs:    http://localhost:8000/api/docs

The backend loads the pre-trained TLSTMFuzzy model at startup (~10-30s). The frontend waits for the
backend health check to pass before becoming available.

- Readiness probe used by Docker Compose: http://localhost:8000/api/health/ready

## Run Locally (Dev mode)

Create `app/.env` with Gemini settings before starting backend:

```bash
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-3.1-flash-lite-preview
GEMINI_API_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
```

### Backend
```bash
cd e:\thesis\app\backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### Frontend (requires Node.js 18+)
```bash
cd e:\thesis\app\frontend
npm install
npm run dev
```

## Pages

| URL                        | Description                          |
|---------------------------|--------------------------------------|
| http://localhost:3000     | Home — hero, stats, rating groups    |
| /methodology              | Methodology — ratios, model pipeline |
| /reports                  | Data table — filter, search, paginate |
| /rating-tool              | AI credit rating tool                 |

## Analyze History Persistence
- Every successful action on `/rating-tool` is now saved as analysis history.
- Saved records appear in `/reports` immediately (newest first).
- Persisted file path: `app/backend/data/prediction_history.csv`.

## Language Toggle
Click the VI / EN button in the navbar to switch between Vietnamese and English.

## AI Explain Provider
- The `/api/explain` endpoint now uses Gemini API directly.
- Required env: `GEMINI_API_KEY`
- Optional env: `GEMINI_MODEL` (default: `gemini-3.1-flash-lite-preview`)
- Optional env: `GEMINI_API_BASE_URL` (default: `https://generativelanguage.googleapis.com/v1beta/openai/`)
- For Docker Compose, frontend can use `API_INTERNAL_URL=http://backend:8000` for server-side calls inside the Docker network.
- Explain input now includes both TLSTM prediction output and structured xAI context (including SHAP-style top feature drivers) for stronger grounding.
- Gemini explain prompt is tuned to respond like a corporate credit-rating analyst and include an indicative S&P-style AAA-D range discussion.
- On `/rating-tool`, `Risk Interpretation` is rendered only when Gemini returns a successful explanation.
