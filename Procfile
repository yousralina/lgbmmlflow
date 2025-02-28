web: uvicorn app:app --host 0.0.0.0 --port=${PORT:-5000}
worker: streamlit run dashboard.py --server.port=$PORT --server.address=0.0.0.0
