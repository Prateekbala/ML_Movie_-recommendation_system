FROM python:3.11-slim

WORKDIR /app


RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*


COPY requirements.txt .


RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir numpy==1.25.2
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir --upgrade numpy==1.25.2


RUN python -c "import numpy; print('NumPy version:', numpy.__version__)"


RUN python -m nltk.downloader punkt stopwords wordnet

COPY main.py .
COPY model/movies_df.pkl ./model/
COPY model/similarity_matrix.pkl ./model/


EXPOSE 8000


CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]