#!/usr/bin/env bash
set -e

# Start the FastAPI REST API in the background (internal port 8000)
uvicorn api.main:app --host 0.0.0.0 --port 8000 &

# Start the Streamlit dashboard in the foreground on the HF Spaces port
streamlit run app/streamlit_app.py \
  --server.port "${PORT:-7860}" \
  --server.address 0.0.0.0 \
  --server.headless true \
  --browser.gatherUsageStats false
