<<<<<<< HEAD
web: uvicorn app:app --host=0.0.0.0 --port=${PORT}
=======
web: uvicorn app:app --host 0.0.0.0 --port=${PORT:-5000}
>>>>>>> 700026dda469e47602bcab2aaddd38a150f3bef3
worker: streamlit run dashboard.py --server.port=$PORT --server.address=0.0.0.0
