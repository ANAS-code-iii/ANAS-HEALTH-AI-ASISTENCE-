"""
health_mcp.py - the MCP server: ML screening + tools that wrap llm.py.

Tools: ask_health_question, list_diseases, predict_disease, health_agent

Run standalone:
    python health_mcp.py            # stdio
    python health_mcp.py --http     # http://127.0.0.1:8001/mcp

NOTE: never name this file `mcp.py` (it shadows the real `mcp` package).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import joblib
import pandas as pd
from fastmcp import FastMCP
mcp = FastMCP("bd-health-assistant")
from llm import answer_question, run_agent

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

MODELS_DIR = Path(
    os.getenv("MODELS_DIR", str(PROJECT_DIR / "ml"))
)

# --------------------------------------------------------------------------- #
# ML DISEASE REGISTRY
# Each disease = <id>_model.pkl + <id>_columns.pkl inside ./saved_models
#   select -> INDEX of the chosen option is sent to the model (LabelEncoder order)
#   number -> numeric input with min / max
# VERIFY: option ORDER is an assumption (alphabetical). Re-order if your
# notebook encoded differently. To add a disease, add one entry below.
# --------------------------------------------------------------------------- #
YES_NO = ["No", "Yes"]

REGISTRY: dict[str, dict] = {
    "cvd": {
        "name": "Cardiovascular Disease (Heart)",
        "description": "Heart disease / heart-attack risk screening.",
        # VERIFY: model has 3 classes (0,1,2); assumed Low / Moderate / High.
        "class_labels": ["Low risk", "Moderate risk", "High risk"],
        "positive_classes": [1, 2],  # classes that count as "YES"
        "fields": {
            "Sex": {"label": "Sex", "type": "select", "options": ["Female", "Male"]},
            "Smoking Status": {"label": "Do you smoke?", "type": "select", "options": YES_NO},
            "Diabetes Status": {"label": "Do you have diabetes?", "type": "select", "options": YES_NO},
            "Physical Activity Level": {
                "label": "Physical activity level",
                "type": "select",
                "options": ["High", "Low", "Moderate"],
            },
            "Family History of CVD": {
                "label": "Heart disease in your family?",
                "type": "select",
                "options": YES_NO,
            },
            "Systolic_BP": {"label": "Systolic blood pressure (upper)", "type": "number",
                            "unit": "mmHg", "min": 70, "max": 250},
            "Diastolic_BP": {"label": "Diastolic blood pressure (lower)", "type": "number",
                             "unit": "mmHg", "min": 40, "max": 150},
        },
    },
    "kidney": {
        "name": "Chronic Kidney Disease",
        "description": "Kidney disease screening from blood and urine test values.",
        # VERIFY: assumed class 1 = CKD present.
        "class_labels": ["No kidney disease", "Kidney disease"],
        "positive_classes": [1],
        "fields": {
            "age": {"label": "Age", "type": "number", "unit": "years", "min": 1, "max": 110},
            "bp": {"label": "Blood pressure (diastolic)", "type": "number", "unit": "mmHg", "min": 40, "max": 200},
            "al": {"label": "Albumin in urine", "type": "number", "unit": "0-5 scale", "min": 0, "max": 5},
            "su": {"label": "Sugar in urine", "type": "number", "unit": "0-5 scale", "min": 0, "max": 5},
            "rbc": {"label": "Red blood cells (urine)", "type": "select", "options": ["Abnormal", "Normal"]},
            "pc": {"label": "Pus cells (urine)", "type": "select", "options": ["Abnormal", "Normal"]},
            "pcc": {"label": "Pus cell clumps", "type": "select", "options": ["Not present", "Present"]},
            "ba": {"label": "Bacteria in urine", "type": "select", "options": ["Not present", "Present"]},
            "bgr": {"label": "Random blood glucose", "type": "number", "unit": "mg/dL", "min": 20, "max": 600},
            "bu": {"label": "Blood urea", "type": "number", "unit": "mg/dL", "min": 1, "max": 400},
            "sc": {"label": "Serum creatinine", "type": "number", "unit": "mg/dL", "min": 0.1, "max": 80},
            "hemo": {"label": "Haemoglobin", "type": "number", "unit": "g/dL", "min": 2, "max": 20},
            "pcv": {"label": "Packed cell volume", "type": "number", "unit": "%", "min": 5, "max": 65},
            "rc": {"label": "Red blood cell count", "type": "number", "unit": "million/cmm", "min": 1, "max": 9},
            "htn": {"label": "Hypertension?", "type": "select", "options": YES_NO},
            "dm": {"label": "Diabetes mellitus?", "type": "select", "options": YES_NO},
            "cad": {"label": "Coronary artery disease?", "type": "select", "options": YES_NO},
            "appet": {"label": "Appetite", "type": "select", "options": ["Good", "Poor"]},
            "pe": {"label": "Swelling in feet (pedal oedema)?", "type": "select", "options": YES_NO},
            "ane": {"label": "Anaemia?", "type": "select", "options": YES_NO},
        },
    },
    "stroke": {
        "name": "Stroke",
        "description": "Stroke risk screening.",
        "class_labels": ["Low stroke risk", "High stroke risk"],
        "positive_classes": [1],
        "fields": {
            "gender": {"label": "Gender", "type": "select", "options": ["Female", "Male", "Other"]},
            "age": {"label": "Age", "type": "number", "unit": "years", "min": 1, "max": 110},
            "hypertension": {"label": "High blood pressure?", "type": "select", "options": YES_NO},
            "heart_disease": {"label": "Heart disease?", "type": "select", "options": YES_NO},
            "ever_married": {"label": "Ever married?", "type": "select", "options": YES_NO},
            "work_type": {
                "label": "Work type",
                "type": "select",
                # alphabetical: Govt_job, Never_worked, Private, Self-employed, children
                "options": ["Government job", "Never worked", "Private job", "Self-employed", "Child"],
            },
            "Residence_type": {"label": "Residence", "type": "select", "options": ["Rural", "Urban"]},
            "avg_glucose_level": {"label": "Average blood glucose", "type": "number",
                                  "unit": "mg/dL (mmol/L x 18)", "min": 40, "max": 500},
            "bmi": {"label": "BMI", "type": "number", "unit": "kg/m²", "min": 10, "max": 70},
            "smoking_status": {
                "label": "Smoking status",
                "type": "select",
                # alphabetical: Unknown, formerly smoked, never smoked, smokes
                "options": ["Unknown", "Formerly smoked", "Never smoked", "Smokes"],
            },
        },
    },
    # "diabetes": {...},    # <- add your 4th model here
    # "<fifth>":  {...},    # <- and your 5th
}


def _load_models() -> None:
    """Load every model that exists on disk.
    Accepts <id>_model.pkl OR <id>.pkl. Column list comes from <id>_columns.pkl,
    or (if that file is missing) from the model's own feature_names_in_."""
    for did, cfg in REGISTRY.items():
        cfg["model"], cfg["columns"], cfg["error"] = None, None, None
        model_p = next((p for p in (MODELS_DIR / f"{did}_model.pkl", MODELS_DIR / f"{did}.pkl")
                        if p.exists()), None)
        cols_p = MODELS_DIR / f"{did}_columns.pkl"
        if model_p is None:
            cfg["error"] = f"{did}_model.pkl (or {did}.pkl) not found"
            if cols_p.exists():
                cfg["columns"] = list(joblib.load(cols_p))
            continue
        try:
            model = joblib.load(model_p)
        except Exception as exc:  # version mismatch, corrupt file ...
            cfg["error"] = f"could not load model: {exc}"
            continue
        if cols_p.exists():
            cols = list(joblib.load(cols_p))
        elif hasattr(model, "feature_names_in_"):
            cols = [str(c) for c in model.feature_names_in_]
        else:
            cfg["error"] = f"{cols_p.name} not found"
            continue
        missing = [c for c in cols if c not in cfg["fields"]]
        if missing:
            cfg["error"] = f"form fields missing for columns: {missing}"
            continue
        cfg["columns"], cfg["model"] = cols, model


_load_models()


def _public_spec(did: str) -> dict:
    cfg = REGISTRY[did]
    return {
        "id": did,
        "name": cfg["name"],
        "description": cfg["description"],
        "ready": cfg["model"] is not None,
        "error": cfg["error"],
        "fields": [
            {"key": c, **cfg["fields"][c]} for c in (cfg["columns"] or cfg["fields"].keys())
        ],
    }


def _encode(cfg: dict, inputs: dict) -> pd.DataFrame:
    row = {}
    for col in cfg["columns"]:
        spec = cfg["fields"][col]
        val = inputs.get(col)
        if val is None or val == "":
            raise ValueError(f"Please fill in: {spec['label']}")
        if spec["type"] == "select":
            if val not in spec["options"]:
                raise ValueError(f"'{val}' is not a valid choice for {spec['label']}")
            row[col] = spec["options"].index(val)
        else:
            try:
                num = float(val)
            except (TypeError, ValueError):
                raise ValueError(f"{spec['label']} must be a number")
            if not spec["min"] <= num <= spec["max"]:
                raise ValueError(f"{spec['label']} must be between {spec['min']} and {spec['max']}")
            row[col] = num
    return pd.DataFrame([row], columns=cfg["columns"])


# --------------------------------------------------------------------------- #
# MCP TOOLS
# --------------------------------------------------------------------------- #
@mcp.tool
def ask_health_question(question: str, history: list[dict] | None = None) -> dict:
    """Answer a general health question. `history` is a list of
    {"role": "user"|"assistant", "content": "..."} from earlier in the chat."""
    return answer_question(question, history)


@mcp.tool
def list_diseases() -> list[dict]:
    """List the disease-screening models and the input fields each one needs."""
    return [_public_spec(d) for d in REGISTRY]


@mcp.tool
def predict_disease(disease: str, inputs: dict) -> dict:
    """Run one ML screening model. Returns a YES/NO verdict plus probabilities."""
    cfg = REGISTRY.get(disease)
    if cfg is None:
        raise ValueError(f"Unknown disease '{disease}'. Options: {list(REGISTRY)}")
    if cfg["model"] is None:
        raise ValueError(f"The {cfg['name']} model is not available yet ({cfg['error']}).")

    X = _encode(cfg, inputs)
    model = cfg["model"]
    proba = model.predict_proba(X)[0]
    classes = [int(c) for c in model.classes_]
    # argmax, but on a tie prefer the riskier (higher) class - safer for screening
    best = max(range(len(classes)), key=lambda i: (round(float(proba[i]), 6), classes[i]))
    pred = classes[best]
    labels = cfg["class_labels"]
    name_of = lambda c: labels[c] if c < len(labels) else str(c)

    return {
        "disease": disease,
        "name": cfg["name"],
        "verdict": "YES" if pred in cfg["positive_classes"] else "NO",
        "result_label": name_of(pred),
        "confidence_percent": round(float(proba[best]) * 100, 1),
        "probabilities": {name_of(c): round(float(p) * 100, 1) for c, p in zip(classes, proba)},
        "inputs": inputs,
    }


@mcp.tool
def health_agent(
    history_and_symptoms: str,
    additional_information: str = "",
    predictions: list[dict] | None = None,
) -> dict:
    """AI agent: reads symptoms/history (+ any ML screening results) and returns
    a structured health analysis with urgency, next steps and specialist advice."""
    return run_agent(history_and_symptoms, additional_information, predictions)


if __name__ == "__main__":
    if "--http" in sys.argv:
        mcp.run(transport="http", host="127.0.0.1", port=8001)
    else:
        mcp.run()