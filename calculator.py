"""Prediction and input validation for the S08 research prototype."""
import json
import math
from numbers import Real
from pathlib import Path

import numpy as np
import xgboost as xgb

APP_DIR = Path(__file__).resolve().parent
ENDPOINTS = ("OS", "CSS")

def load_models(directory=APP_DIR):
    try:
        metadata = json.loads((directory / "model_metadata.json").read_text(encoding="utf-8"))
        if metadata["version"] != "s08-web-v1":
            raise ValueError("Unsupported calculator configuration version.")
        models = {}
        for endpoint in ENDPOINTS:
            spec = metadata["endpoints"][endpoint]
            if len(spec["features"]) != 15 or set(spec["features"]) != (
                set(metadata["numeric"]) | set(spec["categorical"])
            ):
                raise ValueError(f"{endpoint} feature configuration is invalid.")
            for key in ("intercept", "coefficient"):
                if not math.isfinite(spec["parameters"][key]):
                    raise ValueError(f"{endpoint} calibration parameter is invalid.")
            lower, upper = spec["thresholds"]
            if not 0 < lower < upper < 1:
                raise ValueError(f"{endpoint} thresholds are invalid.")
            booster = xgb.Booster(params={"nthread": 1})
            booster.load_model(directory / spec["model"])
            booster.set_param({"nthread": 1, "device": "cpu"})
            start, stop = spec["iteration_range"]
            if start != 0 or not 0 < stop <= booster.num_boosted_rounds():
                raise ValueError(f"{endpoint} boosting range is invalid.")
            if booster.num_features() != len(spec["features"]):
                raise ValueError(f"{endpoint} model feature count differs from the configuration.")
            models[endpoint] = booster
    except (OSError, ValueError, KeyError, TypeError, AttributeError, xgb.core.XGBoostError) as error:
        raise ValueError("Calculator assets could not be loaded. Check the model files and configuration.") from error
    return metadata, models

def encode_input(fields, spec, numeric):
    encoded = []
    for feature in spec["features"]:
        value = fields.get(feature)
        if value is None:
            raise ValueError(f"Enter a value for {feature}.")
        if feature in numeric:
            low, high = numeric[feature]
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError(f"{feature} must be a finite number.")
            if not low <= value <= high:
                raise ValueError(f"{feature} must be between {low} and {high}.")
            if feature in ("Days to treatment", "Age") and value != int(value):
                raise ValueError(f"{feature} must be a whole number.")
        else:
            mapping = spec["categorical"][feature]
            if not isinstance(value, str):
                raise ValueError(f"Select a valid category for {feature}.")
            category = spec["aliases"].get(feature, {}).get(value, value)
            if category not in mapping:
                raise ValueError(f"Select a valid category for {feature}.")
            value = mapping[category]
        encoded.append(value)
    return encoded

def predict_raw(booster, values, spec):
    matrix = xgb.DMatrix(np.asarray([values], dtype=np.float32), nthread=1)
    return float(booster.predict(matrix, iteration_range=tuple(spec["iteration_range"]))[0])

def validate_probability(value):
    if isinstance(value, bool) or not isinstance(value, Real) or not 0 <= value <= 1:
        raise ValueError("Prediction probability must be finite and between 0 and 1.")

def calibrate(p_raw, spec):
    validate_probability(p_raw)
    parameters = spec["parameters"]
    linear = parameters["intercept"] + parameters["coefficient"] * p_raw
    if not math.isfinite(linear):
        raise ValueError("The calibration calculation returned a non-finite value.")
    if linear >= 0:
        return 1 / (1 + math.exp(-linear))
    odds = math.exp(linear)
    return odds / (1 + odds)

def assign_risk(probability, spec):
    validate_probability(probability)
    lower, upper = spec["thresholds"]
    if probability < lower:
        return "Low"
    if probability < upper:
        return "Intermediate"
    return "High"

def predict(fields, metadata, models):
    results = {}
    for endpoint in ENDPOINTS:
        spec = metadata["endpoints"][endpoint]
        values = encode_input(fields, spec, metadata["numeric"])
        raw = predict_raw(models[endpoint], values, spec)
        calibrated = calibrate(raw, spec)
        results[endpoint] = {
            "p_raw": raw,
            "p_cal": calibrated,
            "complement": 1 - calibrated,
            "risk": assign_risk(calibrated, spec),
        }
    return results
