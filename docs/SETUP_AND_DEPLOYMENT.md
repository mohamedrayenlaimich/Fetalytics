# Fetalytics — Setup, Execution, and Deployment Guide

> **Clinical Decision-Support Disclaimer**: Fetalytics is an AI-assisted research and clinical decision-support system. It is not an autonomous diagnostic system. All outputs, contours, and biometric measurements must be reviewed and verified by a qualified healthcare professional.

---

## 1. Environment Configuration

1. Copy `.env.example` to `.env` in the repository root:
   ```powershell
   Copy-Item .env.example .env
   ```
2. Configure the required environment variables in `.env`:
   - `DATABASE_URL`: Neon PostgreSQL connection string (`postgresql://...`)
   - `B2_KEY_ID`, `B2_KEY_NAME`, `B2_APPLICATION_KEY`, `B2_ENDPOINT_URL`, `B2_BUCKET_NAME`
   - `API_KEY_SECRET`, `JWT_SECRET_KEY`, `MAX_UPLOAD_SIZE_MB`, `RATE_LIMIT_PER_MINUTE`
3. Install Python dependencies:
   ```powershell
   python -m pip install -r requirements.txt
   ```

---

## 2. Database Schema & Migrations (Neon PostgreSQL)

Schema migrations are managed via Alembic (`alembic.ini` and `Backend/alembic/`) and idempotently synchronized via `Backend/app/core/migrations.py`.

To run Alembic migrations manually:
```powershell
python -m alembic upgrade head
```

---

## 3. Dataset Ingestion Pipeline (Backblaze B2 → Controlled Storage → Neon PostgreSQL)

To inspect and ingest an authorized dataset from Backblaze B2 (`fetalytics-videos`):

- **Dry-run mode** (validates archives, checks corruption/duplicates/leakage without writing to DB):
  ```powershell
  python AI/datasets/cli.py --bucket fetalytics-videos --b2-key datasets/DatasetV3.zip --subset val --dry-run
  ```
- **Full ingestion into Neon PostgreSQL**:
  ```powershell
  python AI/datasets/cli.py --bucket fetalytics-videos --b2-key datasets/DatasetV3.zip --subset val
  ```

---

## 4. Reproducible Model Training & Held-Out Evaluation

Train `FetalHeadUNet` with deterministic random seeds, subject-level train/val/test separation, baseline Otsu/Ellipse comparison, B2 artifact upload, and PostgreSQL `TrainingRun` / `ModelVersion` registration:

```powershell
python AI/training/cli.py --epochs 16 --batch-size 8 --seed 42
```

Evaluation reports (Dice, IoU, 95% bootstrap confidence intervals, MAE/RMSE, Bland-Altman agreement, Standard-Plane sensitivity/specificity/F1/Brier score, and subgroup analyses) are saved to `results/reports/training_evaluation_report.json`.

---

## 5. Image & Video Inference CLI

- **Single Ultrasound Image (uncalibrated or calibrated)**:
  ```powershell
  python AI/analysis/cli.py --input AI/datasets/extracted/frames/test_10_80_frame_80.png --pixel-spacing-mm 0.25
  ```
- **Ultrasound Video Sequence**:
  ```powershell
  python AI/analysis/cli.py --input tmp_ingest/DatasetV3/extracted/val/videos/test_10_80.avi --frame-step 4
  ```

---

## 6. Running the FastAPI Backend & React Frontend

### Backend (FastAPI)
```powershell
python -m uvicorn Backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```
- Interactive OpenAPI Docs: `http://127.0.0.1:8000/docs`
- Health Check: `http://127.0.0.1:8000/health`

### Frontend (React + TypeScript + Vite)
```powershell
cd Frontend
npm install
npm run dev
```

---

## 7. Running the Automated Test Suite

```powershell
python -m pytest tests/ Backend/test_api_endpoints.py -v
```
