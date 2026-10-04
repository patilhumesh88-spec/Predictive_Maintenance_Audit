import numpy as np
import pandas as pd

COLS = ["Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]",
        "Torque [Nm]", "Tool wear [min]", "Type"]


def row(air=300.0, proc=310.0, rpm=1500, torque=40.0, wear=50, typ="M"):
    return {"Air temperature [K]": air, "Process temperature [K]": proc,
            "Rotational speed [rpm]": rpm, "Torque [Nm]": torque,
            "Tool wear [min]": wear, "Type": typ}


def frame(*rows):
    return pd.DataFrame(list(rows), columns=COLS)


def synthetic(n=600, seed=0):
    """Small synthetic frame with the AI4I schema and a rule-driven label (for model fixtures)."""
    from src.rules import rule_flags
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "Air temperature [K]": rng.normal(300, 2, n).round(1),
        "Process temperature [K]": rng.normal(310, 1.5, n).round(1),
        "Rotational speed [rpm]": rng.normal(1540, 180, n).astype(int).clip(1200, 2800),
        "Torque [Nm]": rng.normal(40, 10, n).clip(4, 75).round(1),
        "Tool wear [min]": rng.integers(0, 250, n),
        "Type": rng.choice(["L", "M", "H"], n, p=[0.6, 0.3, 0.1]),
    })
    df["Machine failure"] = rule_flags(df)["rule_any"].astype(int)
    return df
