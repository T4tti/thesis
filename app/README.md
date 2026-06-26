# VN-Rating — Quick Start Guide

## Prerequisites
- Docker Desktop installed and running
- At least 4GB RAM available for Docker

## Run with Docker Compose (Production-like)

Create a local Compose env file first:

```bash
cd e:\thesis\app
copy .env.example .env
```

Then fill `GEMINI_API_KEY` in `app\.env` if you want Gemini explanations.
The backend still returns deterministic fallback explanations when Gemini is not configured or unavailable.

```bash
cd e:\thesis\app
docker compose -f docker-compose.yml up --build
```

First build may take **5-8 minutes** (installs Python deps + Node deps + compiles Next.js).
The backend image is built from the repository root with a narrow `.dockerignore`, then copies only:

- `app/backend/`
- DMF/DCS model artifact and GraphSAGE reference predictions
- DMF SHAP/LIME CSV artifacts needed by `/api/explain`

- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs:    http://localhost:8000/api/docs

The backend loads the pre-trained DMF/DCS T-LSTM + GraphSAGE runtime at startup (~10-30s). The frontend waits for the
backend health check to pass before becoming available.

- Readiness probe used by Docker Compose: http://localhost:8000/api/health/ready

## Run with Docker Compose (Dev Hot Reload)

The default Compose command also reads `docker-compose.override.yml`, which enables bind mounts and hot reload:

```bash
cd e:\thesis\app
docker compose up --build
```

Use this for local development. It mounts:

- `app/backend` to `/workspace/app/backend`
- `artifacts` read-only to `/workspace/artifacts`
- frontend source, `.next`, and `node_modules` volumes

## Run Locally Without Docker

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
- If PostgreSQL is unavailable, `/api/history` falls back to CSV and returns `history_storage_mode=csv_fallback`.

## Language Toggle
Click the VI / EN button in the navbar to switch between Vietnamese and English.

## AI Explain Provider
- The `/api/explain` endpoint uses Gemini when available.
- Optional env: `GEMINI_API_KEY`
- Optional env: `GEMINI_MODEL` (default: `gemini-3.1-flash-lite-preview`)
- Optional env: `GEMINI_API_BASE_URL` (default: `https://generativelanguage.googleapis.com/v1beta/openai/`)
- For Docker Compose, frontend can use `API_INTERNAL_URL=http://backend:8000` for server-side calls inside the Docker network.
- Explain input includes DMF/DCS context, TLSTM anchor, GraphSAGE runtime metadata, and artifact-backed xAI context.
- If Gemini key/network/quota fails, backend returns HTTP 200 with deterministic fallback explanation instead of breaking the UI.
