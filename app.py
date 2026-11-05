"""Run with: python -m streamlit run app.py"""
import streamlit as st
from xgboost.core import XGBoostError

from calculator import ENDPOINTS, load_models, predict

st.set_page_config(page_title="8-year prostate cancer prediction", layout="wide")
st.title("8-year prostate cancer prediction")
st.subheader("Research prototype")
st.markdown(
    "For retrospective research in men aged 65 or older with prostate cancer "
    "and no surgery at the primary site. The 8-year target starts at diagnosis. "
    "Use recorded treatment information and a known diagnosis-to-treatment interval."
)
st.caption("Changing a treatment input changes the model prediction; it does not estimate treatment benefit.")
st.caption(
    "Before clinical use, evaluate discrimination and calibration in an independent "
    "target population and recalibrate if needed."
)

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
    confirmed = st.checkbox(
        "I confirm that the record is for a man aged 65 or older at diagnosis with prostate cancer "
        "and no primary-site surgery, and that the treatment fields and delay are available as recorded."
    )
    submitted = st.form_submit_button("Calculate 8-year mortality probabilities", type="primary")

if submitted:
    if not confirmed:
        st.error("Confirm population eligibility and availability of treatment information.")
        st.stop()
    try:
        results = predict(fields, metadata, models)
    except ValueError as error:
        st.error(str(error))
        st.stop()
    except XGBoostError:
        st.error("The model could not calculate the prediction. Check the calculator installation.")
        st.stop()
    st.header("Research prototype results")
    labels = {
        "OS": ("All-cause mortality", "Overall survival in the selected OS cohort"),
        "CSS": ("Prostate cancer mortality label", "Probability of the negative CSS label (1 − p)"),
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
                    "These probabilities describe the selected binary CSS cohort, which excluded "
                    "deaths from other causes within 8 years. The CSS output is specific to that cohort."
                )
    st.caption(
        "Risk strata use the unrounded mortality probability and prespecified cutoffs "
        "(OS 30% and 60%, CSS 20% and 50%). They describe model predictions."
    )
st.caption(f"Version {metadata['version']}")
