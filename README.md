# Streamlit Starter App

## Run locally
```bash
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows:
# .venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Project layout
- app.py : landing page
- pages/1_Data_Upload.py : upload + preview + stats
- pages/2_Plotting.py : interactive plots
- utils.py : cached CSV loader

Tip: Upload a CSV in the Data Upload page first.
