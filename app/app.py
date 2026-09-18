from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from predictive_maintenance.config import MODEL_PATH
from predictive_maintenance.predict import load_bundle, predict_record


st.set_page_config(page_title="Machine Failure Risk", page_icon="🛠️", layout="centered")
st.title("Industrial Predictive Maintenance")
st.caption("Cost-sensitive machine-failure risk estimate using the AI4I 2020 dataset")

if not MODEL_PATH.exists():
    st.error("Model artifact not found. Run `python run_pipeline.py` first.")
    st.stop()

bundle = load_bundle(MODEL_PATH)

with st.form("risk_form"):
    product_type = st.selectbox("Product quality type", ["L", "M", "H"])
    air_temperature = st.number_input("Air temperature (K)", 290.0, 310.0, 300.0, 0.1)
    process_temperature = st.number_input("Process temperature (K)", 300.0, 320.0, 310.0, 0.1)
    rotational_speed = st.number_input("Rotational speed (rpm)", 900, 3000, 1500, 1)
    torque = st.number_input("Torque (Nm)", 0.0, 100.0, 40.0, 0.1)
    tool_wear = st.number_input("Tool wear (min)", 0, 300, 100, 1)
    submitted = st.form_submit_button("Estimate failure risk")

if submitted:
    result = predict_record(
        {
            "Type": product_type,
            "Air temperature": air_temperature,
            "Process temperature": process_temperature,
            "Rotational speed": rotational_speed,
            "Torque": torque,
            "Tool wear": tool_wear,
        },
        bundle,
    )
    st.metric("Estimated failure risk", f"{result['failure_probability']:.1%}")
    st.write(f"Risk band: **{result['risk_band'].title()}**")
    if result["maintenance_alert"]:
        st.warning("Model alert: inspect the operating condition before continuing production.")
    else:
        st.success("No model alert at the selected cost-sensitive threshold.")
    st.caption(
        f"Decision threshold: {result['decision_threshold']:.2f}. "
        "This demonstration is not a substitute for an engineered maintenance policy."
    )

