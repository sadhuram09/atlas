# ATLAS — Phase 5 Deployment Guide

## What deploys where

| Layer    | Platform | URL pattern                          |
|----------|----------|--------------------------------------|
| Backend  | Railway  | `https://atlas-xxxx.up.railway.app`  |
| Frontend | Vercel   | `https://atlas-frontend.vercel.app`  |

---

## Prerequisites

- GitHub repo with code pushed
- [Railway account](https://railway.app) (free)
- [Vercel account](https://vercel.com) (free)
- Groq API key from [console.groq.com](https://console.groq.com/keys) (free)

---

## Step 1 — Deploy Backend to Railway

### 1a. Create Railway project

1. Go to [railway.app](https://railway.app) → **New Project**
2. Click **Deploy from GitHub repo** → select your repo
3. Railway auto-detects the `Dockerfile` ✓

### 1b. Set environment variables

In Railway dashboard → your service → **Variables**, add:

```
GROQ_API_KEY=gsk_your_actual_key_here
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO
CORS_ORIGINS=["https://your-vercel-url.vercel.app"]
```

> ⚠️ Copy the Railway service URL after first deploy — you'll need it for `CORS_ORIGINS`

### 1c. Get Railway secrets for GitHub Actions

1. Railway dashboard → **Account Settings** → **Tokens** → **Create token**
2. Copy the token → add as `RAILWAY_TOKEN` in GitHub Secrets
3. Get Service ID from Railway URL: `railway.app/project/xxx/service/SERVICE_ID_HERE`
4. Add as `RAILWAY_SERVICE` in GitHub Secrets

---

## Step 2 — Deploy Frontend to Vercel

### 2a. First deploy (manual)

```bash
npm install -g vercel
cd atlas   # the folder with vercel.json and your HTML files
vercel --prod
```

Follow the prompts — Vercel will detect it as a static site.

### 2b. Get Vercel secrets for GitHub Actions

```bash
vercel whoami          # get your username/org
vercel project ls      # get project name
```

Then from [vercel.com/account/tokens](https://vercel.com/account/tokens):

Add these to GitHub Secrets:
- `VERCEL_TOKEN` → from Vercel dashboard
- `VERCEL_ORG_ID` → Settings → General → Your ID  
- `VERCEL_PROJECT_ID` → Project → Settings → General → Project ID

---

## Step 3 — Set GitHub Secrets

Go to your repo → **Settings** → **Secrets and variables** → **Actions** → **New repository secret**

Add all 5 secrets:
```
RAILWAY_TOKEN
RAILWAY_SERVICE
VERCEL_TOKEN
VERCEL_ORG_ID
VERCEL_PROJECT_ID
```

---

## Step 4 — Trigger CI/CD

```bash
git add .
git commit -m "feat: phase 5 deployment"
git push origin main
```

GitHub Actions will:
1. Run lint + tests
2. Deploy backend to Railway (parallel)
3. Deploy frontend to Vercel (parallel)

Check progress at: `github.com/YOUR_USERNAME/YOUR_REPO/actions`

---

## Step 5 — Verify Live Deployment

```bash
# Backend health check
curl https://your-railway-url.up.railway.app/health

# Expected response:
# {"status":"healthy","version":"0.1.0","environment":"production"}
```

Open your Vercel URL in the browser — the landing page should load.

---

## Local dev (quick reference)

```bash
cd atlas
cp .env.example .env
# Edit .env — add your GROQ_API_KEY

pip install poetry
poetry install
poetry run python main.py

# Server at http://localhost:8000
# API docs at http://localhost:8000/docs
```

---

## Environment variables reference

| Variable            | Required | Default       | Description                        |
|---------------------|----------|---------------|------------------------------------|
| `GROQ_API_KEY`      | ✅ YES   | placeholder   | Get free at console.groq.com       |
| `ENVIRONMENT`       | ✅ YES   | development   | production / staging / development |
| `DEBUG`             | no       | false         | Enables debug logging              |
| `LOG_LEVEL`         | no       | INFO          | DEBUG / INFO / WARNING / ERROR     |
| `CORS_ORIGINS`      | ✅ YES   | localhost     | JSON array of allowed origins      |
| `PORT`              | no       | 8000          | Railway injects this automatically |
| `DATABASE_URL`      | no       | local pg      | asyncpg connection string          |
| `LANGSMITH_API_KEY` | no       | placeholder   | For LangSmith tracing              |

---

## Architecture

```
GitHub Actions CI/CD
       │
       ├── push to main
       │
       ├── [test] ruff + mypy + pytest
       │
       ├── [deploy-backend] → Railway
       │       └── Dockerfile (multi-stage, python:3.12-slim)
       │               └── FastAPI + uvicorn on $PORT
       │                       └── /health  → health check
       │                       └── /tasks   → task submission
       │                       └── /ws/{id} → WebSocket live feed
       │
       └── [deploy-frontend] → Vercel
               └── vercel.json (static HTML)
                       └── /        → landing page
                       └── /dashboard → ATLAS dashboard
```
