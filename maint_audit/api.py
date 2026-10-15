"""FastAPI inference service for failure risk scoring."""
from typing import List, Optional, Literal
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict
import pandas as pd
import joblib

from .config import (
    MODEL_PATH, PRODUCTION_THRESHOLD,
    AIR, PROC, RPM, TORQUE, WEAR, TYPE
)
from .predict import predict_frame

bundle = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global bundle
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model artifact not found at {MODEL_PATH}")
    bundle = joblib.load(MODEL_PATH)
    yield


app = FastAPI(
    title="Predictive Maintenance Failure Risk API",
    description="Leakage-aware risk scoring and physical failure rule diagnostics for industrial milling telemetry.",
    version="1.0.0",
    lifespan=lifespan,
)


class TelemetryRecord(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    air_temperature_k: float = Field(..., ge=250.0, le=400.0, alias="Air temperature [K]", description="Air temperature in Kelvin")
    process_temperature_k: float = Field(..., ge=250.0, le=450.0, alias="Process temperature [K]", description="Process temperature in Kelvin")
    rotational_speed_rpm: float = Field(..., ge=1.0, le=10000.0, alias="Rotational speed [rpm]", description="Spindle speed in RPM")
    torque_nm: float = Field(..., ge=0.0, le=500.0, alias="Torque [Nm]", description="Torque in Nm")
    tool_wear_min: float = Field(..., ge=0.0, le=1000.0, alias="Tool wear [min]", description="Cumulative tool wear in minutes")
    machine_type: Literal["L", "M", "H"] = Field(..., alias="Type", description="Product variant quality (L, M, H)")
    udi: Optional[int] = Field(None, alias="UDI")
    product_id: Optional[str] = Field(None, alias="Product ID")


class PredictionResponse(BaseModel):
    rule_hdf: bool
    rule_pwf: bool
    rule_osf: bool
    rule_any: bool
    ml_risk: float
    risk_flag: bool
    likely_mode: str
    status: str
    udi: Optional[int] = None
    product_id: Optional[str] = None


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": bundle is not None,
        "model_name": bundle.get("metadata", {}).get("model_name", "RandomForest") if bundle else None,
        "threshold": bundle.get("threshold", PRODUCTION_THRESHOLD) if bundle else PRODUCTION_THRESHOLD,
    }


def _record_to_df(records: List[TelemetryRecord]) -> pd.DataFrame:
    rows = []
    for r in records:
        d = {
            AIR: r.air_temperature_k,
            PROC: r.process_temperature_k,
            RPM: r.rotational_speed_rpm,
            TORQUE: r.torque_nm,
            WEAR: r.tool_wear_min,
            TYPE: r.machine_type,
        }
        if r.udi is not None:
            d["UDI"] = r.udi
        if r.product_id is not None:
            d["Product ID"] = r.product_id
        rows.append(d)
    return pd.DataFrame(rows)


@app.post("/predict", response_model=PredictionResponse)
def predict_single(record: TelemetryRecord):
    if bundle is None:
        raise HTTPException(status_code=503, detail="Model bundle not loaded")
    df = _record_to_df([record])
    res = predict_frame(df, bundle)
    row = res.iloc[0]
    return PredictionResponse(
        rule_hdf=bool(row["rule_hdf"]),
        rule_pwf=bool(row["rule_pwf"]),
        rule_osf=bool(row["rule_osf"]),
        rule_any=bool(row["rule_any"]),
        ml_risk=float(row["ml_risk"]),
        risk_flag=bool(row["risk_flag"]),
        likely_mode=str(row["likely_mode"]),
        status=str(row["status"]),
        udi=int(row["UDI"]) if "UDI" in row and pd.notna(row["UDI"]) else None,
        product_id=str(row["Product ID"]) if "Product ID" in row and pd.notna(row["Product ID"]) else None,
    )


@app.post("/predict/batch", response_model=List[PredictionResponse])
def predict_batch(records: List[TelemetryRecord]):
    if bundle is None:
        raise HTTPException(status_code=503, detail="Model bundle not loaded")
    if not records:
        return []
    df = _record_to_df(records)
    res = predict_frame(df, bundle)
    out = []
    for _, row in res.iterrows():
        out.append(
            PredictionResponse(
                rule_hdf=bool(row["rule_hdf"]),
                rule_pwf=bool(row["rule_pwf"]),
                rule_osf=bool(row["rule_osf"]),
                rule_any=bool(row["rule_any"]),
                ml_risk=float(row["ml_risk"]) if pd.notna(row["ml_risk"]) else 0.0,
                risk_flag=bool(row["risk_flag"]) if pd.notna(row["risk_flag"]) else False,
                likely_mode=str(row["likely_mode"]),
                status=str(row["status"]),
                udi=int(row["UDI"]) if "UDI" in row and pd.notna(row["UDI"]) else None,
                product_id=str(row["Product ID"]) if "Product ID" in row and pd.notna(row["Product ID"]) else None,
            )
        )
    return out
