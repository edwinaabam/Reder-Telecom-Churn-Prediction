FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Expose the ports (8000 for API, 8501 for Streamlit)
EXPOSE 8000
EXPOSE 8501

# Start the API in the background and then start Streamlit
CMD uvicorn deploy.app:app --host 0.0.0.0 --port 8000 & python -m streamlit run deploy/streamlit_app.py --server.port 8501 --server.address 0.0.0.0