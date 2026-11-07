"""Run with: python -m streamlit run app.py"""
import streamlit as st
from xgboost.core import XGBoostError

from calculator import ENDPOINTS, load_models, predict

st.set_page_config(page_title="8-year prostate cancer prediction", layout="wide")
st.title("8-year prostate cancer prediction")
st.subheader("Web-based prognostic calculator")
st.markdown("For prognostic research in men aged ≥65 years with prostate cancer managed without primary-site surgery. Enter documented clinical and treatment characteristics. The eight-year horizon begins at diagnosis.")

@st.cache_resource
def load_calculator():
    return load_models()

try:
    metadata, models = load_calculator()
except ValueError as error:
    st.error(str(error))
    st.stop()

helpers = {
    "Days to treatment": "Enter the recorded interval. Use 0 only for treatment on the day of diagnosis.",
    "PSA value": "PSA in ng/mL. Use 0.1 for a recorded PSA of 0.1 or less, and 98 for a recorded PSA of 98 or higher.",
    "Age": "Age at diagnosis. Use 90 for age 90 or older.",
    "Radiation": "Use the original registry coding. No and unknown share a category.",
    "Chemotherapy": "Use the original registry coding. No and unknown share a category.",
}
groups = (
    ("Patient characteristics", ("Age", "PSA value", "Race", "Marital status")),
    ("Tumour characteristics", ("Histological grade", "T stage", "N stage", "Gleason Score",
                               "Bone metastasis", "Liver metastasis", "Lung metastasis", "Brain metastasis")),
    ("Treatment information", ("Days to treatment", "Radiation", "Chemotherapy")),
)
fields = {}
with st.form("patient_inputs"):
    for column, (heading, features) in zip(st.columns(3), groups):
        with column:
            st.subheader(heading)
            for name in features:
                if name in metadata["numeric"]:
                    low, high = metadata["numeric"][name]
                    fields[name] = st.number_input(
                        name, min_value=low, max_value=high, value=None,
                        step=0.1 if name == "PSA value" else 1, help=helpers.get(name),
                    )
                else:
                    categories = metadata["endpoints"]["OS"]["categorical"][name]
                    fields[name] = st.selectbox(name, list(categories), index=None, help=helpers.get(name))
    submitted = st.form_submit_button("Calculate 8-year mortality probabilities", type="primary")

if submitted:
    try:
        results = predict(fields, metadata, models)
    except ValueError as error:
        st.error(str(error))
        st.stop()
    except XGBoostError:
        st.error("The model could not calculate the prediction. Check the calculator installation.")
        st.stop()
    st.header("8-year prognosis")
    labels = {
        "OS": ("8-year all-cause mortality (p)", "8-year overall survival (1 − p)"),
        "CSS": ("8-year prostate cancer mortality (p)", "8-year cancer-specific survival (1 − p)"),
    }
    for column, endpoint in zip(st.columns(2), ENDPOINTS):
        result = results[endpoint]
        death_label, complement_label = labels[endpoint]
        with column:
            st.subheader(endpoint)
            st.metric(death_label, f"{result['p_cal']:.2%}")
            st.metric(complement_label, f"{result['complement']:.2%}")
            st.markdown(f"**Risk stratum: {result['risk']}**")
            if endpoint == "CSS":
                st.caption(
                    "The CSS estimates are conditional on the binary cohort definition, which excluded "
                    "deaths from other causes within eight years."
                )
    st.caption(
        "Risk strata use the unrounded mortality probability and prespecified cutoffs "
        "(OS 30% and 60%, CSS 20% and 50%). They describe model predictions."
    )

