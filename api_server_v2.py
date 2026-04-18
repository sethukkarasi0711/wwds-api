
"""
Wrong-Way Detection — Production FastAPI Microservice v2
Layered Microservice Pipeline:
  Layer 1: GPS Ingest & Preprocessing
  Layer 2: Map Matching (HMM)
  Layer 3: Rule Engine (8 gates)
  Layer 4: ML Ensemble (XGBoost)
  Layer 5: Alert Broadcast (HARMAN Ignite format)
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import pickle, numpy as np, math, json, time

app = FastAPI(
    title="Wrong-Way Detection Microservice",
    description="Production B2B API — Rule + XGBoost Hybrid | HMM Map Matching",
    version="2.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

with open("outputs/models/xgb_model.pkl","rb") as f: XGB_MODEL = pickle.load(f)
with open("outputs/models/scaler.pkl","rb")    as f: SCALER    = pickle.load(f)

FEATURE_COLS = [
    "city_enc","avg_angle","max_angle","pct_high_angle",
    "n_ww_pts","n_uncertain_pts","n_normal_pts",
    "n_intersection","n_twoway","n_lowspeed","n_suppressed",
    "max_confidence","avg_confidence","max_consecutive",
    "avg_speed","max_speed","total_points",
    "max_evidence","avg_evidence","avg_hcr","max_hcr",
]
CITY_ENC = {"Chennai":0,"Bengaluru":1,"Hyderabad":2}

class VehicleFeatures(BaseModel):
    vehicle_id:str; city:str
    avg_angle:float; max_angle:float; pct_high_angle:float
    n_ww_pts:int; n_uncertain_pts:int; n_normal_pts:int
    n_intersection:int; n_twoway:int; n_lowspeed:int; n_suppressed:int
    max_confidence:float; avg_confidence:float; max_consecutive:int
    avg_speed:float; max_speed:float; total_points:int
    max_evidence:float; avg_evidence:float; avg_hcr:float; max_hcr:float

class AlertResponse(BaseModel):
    alert_id:str; vehicle_id:str; city:str
    hybrid_predicted:str; combined_confidence:float
    ml_probability:float; risk_level:str
    recommended_action:str; broadcast_zones:dict
    detection_pipeline:dict; latency_ms:float

@app.get("/health")
def health():
    return {"status":"ok","service":"wwds-v2","version":"2.0.0",
            "cities":list(CITY_ENC.keys()),
            "pipeline":["Kalman","RANSAC","SavitzkyGolay","HMM","Rules8Gate","XGBoost"]}

@app.post("/api/v2/detect", response_model=AlertResponse)
def detect(feat:VehicleFeatures):
    t0  = time.time()
    ce  = CITY_ENC.get(feat.city,0)
    row = np.array([[ce,feat.avg_angle,feat.max_angle,feat.pct_high_angle,
                     feat.n_ww_pts,feat.n_uncertain_pts,feat.n_normal_pts,
                     feat.n_intersection,feat.n_twoway,feat.n_lowspeed,feat.n_suppressed,
                     feat.max_confidence,feat.avg_confidence,feat.max_consecutive,
                     feat.avg_speed,feat.max_speed,feat.total_points,
                     feat.max_evidence,feat.avg_evidence,feat.avg_hcr,feat.max_hcr]])
    mp      = float(XGB_MODEL.predict_proba(SCALER.transform(row))[0,1])
    comb    = round(0.5*feat.max_confidence + 0.5*mp, 3)
    pred    = "WRONG_WAY" if comb>=0.4 else "NORMAL"
    risk    = "HIGH" if comb>0.7 else "MEDIUM" if comb>0.4 else "LOW"
    latency = round((time.time()-t0)*1000, 2)
    return AlertResponse(
        alert_id        = f"HARMAN-{feat.vehicle_id}-{int(time.time())}",
        vehicle_id      = feat.vehicle_id,
        city            = feat.city,
        hybrid_predicted= pred,
        combined_confidence=comb,
        ml_probability  = round(mp,3),
        risk_level      = risk,
        recommended_action="BROADCAST_ALERT_ZONE_A_B_C" if pred=="WRONG_WAY" else "MONITOR",
        broadcast_zones = {"zone_A_m":100,"zone_B_m":300,"zone_C_m":700},
        detection_pipeline={
            "gps":["Kalman","RANSAC","SavitzkyGolay"],
            "matching":"HMM_Viterbi",
            "rules":"8_Gate",
            "ml":"XGBoost_300est",
        },
        latency_ms=latency,
    )

@app.get("/api/v2/stats")
def stats():
    return {
        "cities"     : list(CITY_ENC.keys()),
        "vehicles"   : "500+",
        "model"      : "XGBoost(300) + Rule-8Gate Hybrid",
        "map_matching": "HMM Viterbi",
        "gps_preproc": ["Kalman","RANSAC","Savitzky-Golay"],
        "threshold"  : 0.4,
    }
