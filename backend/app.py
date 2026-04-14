from __future__ import annotations
import json
import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import jwt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

# ── make worker importable ──────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from worker.extract_melt_block import process_file
from worker.resnet18_inference import (
    curve_to_image,
    get_model,
    ml_inference,
    transform,
)

# ── config ──────────────────────────────────────────────────────────────────
PASSCODE   = "wildsense"
SECRET_KEY = "ws-jwt-secret-8f2k9x3p-change-in-prod"
ALGORITHM  = "HS256"

app = FastAPI(title="WildSense API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer()


# ── auth ─────────────────────────────────────────────────────────────────────
class AuthRequest(BaseModel):
    passcode: str


def require_auth(creds: HTTPAuthorizationCredentials = Depends(security)):
    try:
        jwt.decode(creds.credentials, SECRET_KEY, algorithms=[ALGORITHM])
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired token")


@app.post("/api/auth")
async def authenticate(body: AuthRequest):
    if body.passcode != PASSCODE:
        raise HTTPException(status_code=401, detail="Invalid passcode")
    token = jwt.encode(
        {"exp": datetime.now(tz=timezone.utc) + timedelta(hours=24), "iss": "wildsense"},
        SECRET_KEY,
        algorithm=ALGORITHM,
    )
    return {"token": token}


# ── helpers ──────────────────────────────────────────────────────────────────
def _run_wide_inference(df: pd.DataFrame) -> list[dict]:
    """Inference on wide-format CSV (numeric column names = temperatures)."""
    temp_cols = []
    for col in df.columns:
        try:
            float(str(col).strip())
            temp_cols.append(col)
        except (ValueError, TypeError):
            pass

    if len(temp_cols) < 5:
        raise ValueError("No temperature columns found in CSV.")

    temperatures = [float(str(c).strip()) for c in temp_cols]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, idx2label = get_model(device)

    results = []
    for row_idx, (_, row) in enumerate(df.iterrows()):
        try:
            melting = [
                float(row[c]) if pd.notna(row[c]) else 0.0
                for c in temp_cols
            ]
            img    = curve_to_image(temperatures, melting)
            tensor = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                probs = F.softmax(model(tensor), dim=1).cpu().numpy()[0]

            top5 = np.argsort(-probs)[:5]
            results.append({
                "sample_index": row_idx,
                "predictions": [
                    {"rank": int(r), "species": idx2label[int(i)], "confidence": float(probs[i])}
                    for r, i in enumerate(top5, 1)
                ],
                "curve_data": {
                    "temperatures": temperatures[::4],
                    "signal": melting[::4],
                },
            })
        except Exception as e:
            results.append({"sample_index": row_idx, "error": str(e), "predictions": []})
    return results


def _run_long_inference(filepath: str, n_rows: int) -> list[dict]:
    """Inference on long-format CSV with Temperature (°C) / Melting Curve Data columns."""
    results = []
    for i in range(n_rows):
        res = ml_inference(filepath, sample_index=i)
        if res["success"]:
            results.append({
                "sample_index": i,
                "predictions": res["predictions"][:5],
                "curve_data": res.get("curve_data", {}),
            })
        else:
            results.append({"sample_index": i, "error": res.get("error", "Unknown"), "predictions": []})
    return results


# ── analyze endpoint ──────────────────────────────────────────────────────────
@app.post("/api/analyze")
async def analyze(
    file: UploadFile = File(...),
    _: None = Depends(require_auth),
):
    suffix = Path(file.filename or "upload.csv").suffix.lower()
    if suffix != ".csv":
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    processed_path = tmp_path + "_proc.csv"
    used_processed = False

    try:
        # ── early validation: reject files that aren't valid qPCR data ──
        # Strategy 1 – raw qPCR file with Start/End markers
        has_markers = False
        try:
            process_file(tmp_path, processed_path)
            df = pd.read_csv(processed_path)
            used_processed = True
            has_markers = True
        except Exception:
            pass

        if not has_markers:
            try:
                df = pd.read_csv(tmp_path)
            except Exception:
                raise HTTPException(status_code=400, detail="File is not a valid CSV.")

            if df.empty or len(df.columns) < 2:
                raise HTTPException(status_code=400, detail="CSV file is empty or has too few columns.")

            is_long = "Temperature (°C)" in df.columns and "Melting Curve Data" in df.columns

            if not is_long:
                # Check for wide format: columns should be numeric (temperature values)
                numeric_cols = 0
                for col in df.columns:
                    try:
                        float(str(col).strip())
                        numeric_cols += 1
                    except (ValueError, TypeError):
                        pass
                if numeric_cols < 5:
                    raise HTTPException(
                        status_code=400,
                        detail="Unrecognized CSV format. Expected qPCR melting curve data with "
                               "temperature columns (wide format), 'Temperature (°C)' / "
                               "'Melting Curve Data' columns (long format), or raw qPCR "
                               "Start/End markers.",
                    )

        # ── run inference ──
        if has_markers:
            results = _run_wide_inference(df)
        elif is_long:
            results = _run_long_inference(tmp_path, len(df))
        else:
            results = _run_wide_inference(df)

        return {"success": True, "results": results, "total_samples": len(results)}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        if used_processed and os.path.exists(processed_path):
            os.unlink(processed_path)
