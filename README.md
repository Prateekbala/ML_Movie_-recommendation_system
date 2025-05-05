# Movie Recommendation API

This project is a FastAPI-based backend service for recommending movies based on user input. It offers two recommendation strategies: one using traditional cosine similarity on metadata, and another using a sentence transformer model to find semantically similar movies.

Whether you're building a movie discovery app or experimenting with recommendation systems, this API is designed to be easy to use, flexible, and fast.

---

## Features

- Traditional movie recommendations using a precomputed similarity matrix.
- Semantic recommendations powered by Sentence Transformers (`all-MiniLM-L6-v2`).
- Fuzzy matching to handle minor typos or variations in movie titles.
- Ready-to-use REST API with automatic documentation (Swagger).
- CORS support for frontend integration.

---

## Project Structure

.
├── main.py # FastAPI application
├── model/
│ ├── movies_df.pkl # Pickled DataFrame with movie data
│ └── similarity_matrix.pkl# Pickled similarity matrix
├── requirements.txt # Dependencies
└── README.md # Project documentation


## Requirements

- Python 3.8 or higher

### Installing dependencies

```bash
pip install -r requirements.txt
