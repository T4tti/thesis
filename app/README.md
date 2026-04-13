# VN-Rate — Quick Start Guide

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

The backend trains the LightGBM model at startup (~30-60s). The frontend waits for the
backend health check to pass before becoming available.

## Run Locally (Dev mode)

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

## Language Toggle
Click the VI / EN button in the navbar to switch between Vietnamese and English.
