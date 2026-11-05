# S08 research prototype

Version `s08-web-v1`. The calculator uses separate XGBoost web-export weights for OS and CSS. The two endpoint-specific probability estimates use 94 and 82 rounds, respectively, followed by logistic calibration on the raw probability scale. The model and configuration paths are relative to `app.py`.

## Local run

Use Python 3.12 in a virtual environment. From the folder containing `app.py`:

```text
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The input form starts empty. For a synthetic example, enter age 71, PSA 7.5, and 57 days to treatment. Choose White, Married (including common law), Grade I&Grade II, T2, N0, Gleason 7, Yes for Radiation, No/Unknown for Chemotherapy and No for all four metastatic sites. Confirm the eligibility and data-availability checkbox, then calculate. Use synthetic inputs for screenshots.

## Author deployment handoff

Copy the contents of this directory together into the deployment repository. Select Python 3.12 and `app.py` as the Streamlit entry point. Verify that the page footer reports `s08-web-v1` and that the synthetic example displays both mortality outputs, their complements and the two risk strata. Return the URL and screenshot(s) of the input and results views for Supplementary Figure 3, with the tested revision if available. Keep account or repository identifiers out of the screenshot intended for the blinded supplement.

The prototype illustrates fixed-horizon retrospective predictions after treatment information is known. It uses the original selected binary endpoints and their prespecified mortality strata. The primary manuscript evaluation and test-set stratum rates refer to the archived paper predictions. External evaluation and any necessary recalibration require independent data with matching endpoint definitions.
