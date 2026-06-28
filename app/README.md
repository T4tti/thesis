# VN-Rating — Quick Start Guide

VN-Rating is an AI-powered Corporate Credit Rating Platform designed to evaluate corporate credit risk using 12 core financial indicators. The platform delivers instant classifications of Investment Grade (IG), High Yield (HY), or Distressed tiers, accompanied by detailed probability distributions and AI-generated risk explanations (using Gemini).

---

## 📋 Prerequisites
- **Docker Desktop** installed and running.
- At least **4GB RAM** allocated to Docker.
- A **Gemini API Key** (optional, for risk explanation features).

---

## 🛠️ Environment Configuration

Before running the application, initialize the local environment configuration file:

```bash
cd e:\thesis\app
copy .env.example .env
```

Open the newly created `app/.env` and optionally fill in your `GEMINI_API_KEY`:
```env
# Database Settings
POSTGRES_USER=vnrate_user
POSTGRES_PASSWORD=vnrate_password
POSTGRES_DB=vnrate_db

# API Settings
NEXT_PUBLIC_API_URL=http://localhost:8000
API_INTERNAL_URL=http://backend:8000

# xAI Settings (Optional)
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-3.1-flash-lite-preview
GEMINI_API_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
```
*Note: If `GEMINI_API_KEY` is not provided, the backend falls back to deterministic rule-based credit risk explanations.*

---

## 🚀 Running the Application

### Option A: Using Docker Compose (Recommended)

Run the full platform in detached mode:
```bash
cd e:\thesis\app
docker compose up --build -d
```
*First-time builds may take **5 to 8 minutes** to download and compile dependencies.*

#### Service Endpoints:
- **Frontend App:** [http://localhost:3000](http://localhost:3000)
- **Backend API:** [http://localhost:8000](http://localhost:8000)
- **API Documentation:** [http://localhost:8000/api/docs](http://localhost:8000/api/docs)

#### Dev Mode with Hot Reload:
For active development with source bind-mounting, simply run:
```bash
docker compose up --build
```
This reads `docker-compose.override.yml` to support hot reload for both frontend and backend.

---

### Option B: Running Locally (Without Docker)

Ensure you have Python 3.10+ and Node.js 18+ installed on your host.

#### 1. Backend Service
```bash
cd e:\thesis\app\backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

#### 2. Frontend Service
```bash
cd e:\thesis\app\frontend
npm install
npm run dev
```

---

## 🖥️ Platform Pages & Navigation

| URL | Component | Description |
| :--- | :--- | :--- |
| [http://localhost:3000](http://localhost:3000) | **Home** | Overview, platform statistics, and risk category breakdown (IG / HY / Distressed). |
| `/methodology` | **Methodology** | Documentation of 12 core financial ratios, model validation, and pipeline setup. |
| `/rating-tool` | **Rating Tool** | AI-powered interactive form with CSV autofill capability to analyze corporate credit risk. |
| `/reports` | **Reports & Data** | Historical database displaying previous credit ratings with filtering and search features. |

---

## 📊 Core Features

### 1. Credit Rating History Persistence
- Every successful rating assessment completed via `/rating-tool` is saved to history.
- Saved records are immediately queryable in `/reports` (sorted newest first).
- **Storage Mode:** Defaults to PostgreSQL. If database connection is unavailable, it automatically falls back to CSV storage (`app/backend/data/prediction_history.csv`) and reports `history_storage_mode=csv_fallback`.

### 2. Bilingual Support (VI / EN)
- Supports a toggle in the navbar to switch between **Vietnamese** and **English**.
- The Vietnamese terminology has been fully aligned with **"Xếp hạng Tín dụng Doanh nghiệp"** (Corporate Credit Rating) instead of the older "Xếp hạng Tín nhiệm" to ensure professional financial accuracy.

### 3. AI Risk Explainer (Gemini Integration)
- The `/api/explain` endpoint leverages Gemini models to translate ML probability vectors and financial metrics into human-readable risk reports.
- Employs resilient fallback logic: if the API key is missing or request limits are hit, it serves a structured, deterministic template without breaking the user interface.
