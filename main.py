from fastapi import FastAPI, HTTPException, Query
import pandas as pd
import numpy as np
import re
from difflib import get_close_matches
import uvicorn
from pydantic import BaseModel
from typing import List
import pickle
import os
from fastapi.middleware.cors import CORSMiddleware
import torch
from sentence_transformers import SentenceTransformer, util

app = FastAPI(
    title="Movie Recommendation API",
    description="API for recommending similar movies based on user input",
    version="1.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


model_files = {
    "df": "model/movies_df.pkl",
    "similarity": "model/similarity_matrix.pkl"
}


sentence_model = None
embeddings = None

# Response models
class MovieRecommendation(BaseModel):
    title: str
    movie_id: int
    similarity: float

class RecommendationResponse(BaseModel):
    recommendations: List[MovieRecommendation]
    query: str
    matched_query: str


def load_model_data():
    """Load the pre-trained model data from pickle files"""
    try:
        with open(model_files["df"], "rb") as f:
            df = pickle.load(f)
        with open(model_files["similarity"], "rb") as f:
            similarity = pickle.load(f)
        return df, similarity
    except FileNotFoundError:
        raise Exception("Model files not found. Please run the model training script first.")

# Initialize sentence transformer model
def init_sentence_transformer():
    global sentence_model, embeddings
    try:
        df, _ = load_model_data()
        sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        # Pre-compute embeddings for all movies
        sentences = df['combined'].tolist()
        embeddings = sentence_model.encode(sentences, convert_to_tensor=True)
        print("Sentence transformer model initialized successfully!")
    except Exception as e:
        print(f"Error initializing sentence transformer: {e}")
        sentence_model = None
        embeddings = None

# API routes
@app.get("/", response_model=dict)
def read_root():
    """Root endpoint with API information"""
    return {
        "message": "Movie Recommendation API",
        "version": "1.0.0",
        "endpoints": {
            "/recommend": "Get movie recommendations using traditional similarity",
            "/recommend_transformer": "Get movie recommendations using sentence transformer",
            "/movies": "Get list of available movies"
        }
    }

@app.get("/recommend", response_model=RecommendationResponse)
def recommend_movies(movie: str = Query(..., description="Movie title to get recommendations for"), top_k: int = Query(6, description="Number of recommendations")):
    """Get movie recommendations based on input movie title using traditional similarity"""
    df, similarity = load_model_data()
    
    # Preprocess the movie title: remove special characters and lowercase
    movie_processed = re.sub(r'[^\w\s]', '', movie).lower()
    
    # Find closest matching title in dataframe
    all_titles = df['Title'].str.lower().tolist()
    closest_match = get_close_matches(movie_processed, all_titles, n=1, cutoff=0.3)
    
    if closest_match:
        movie_title = closest_match[0]  # Use closest match if found
    else:
        raise HTTPException(status_code=404, detail=f"Movie '{movie}' not found in the dataset.")
    
    # Get movie index
    movie_index = df[df['Title'].str.lower() == movie_title].index[0]
    
    # Calculate similarities
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:top_k+1]
    
    # Format recommendations
    recommendations = []
    for i in movies_list:
        idx = i[0]
        recommendations.append(MovieRecommendation(
            title=df.iloc[idx].Title,
            movie_id=int(df.iloc[idx].movie_id),
            similarity=float(i[1])
        ))
    
    return RecommendationResponse(
        recommendations=recommendations,
        query=movie,
        matched_query=df.iloc[movie_index].Title
    )

@app.get("/recommend_transformer", response_model=RecommendationResponse)
def recommend_movies_transformer(movie: str = Query(..., description="Movie title to get recommendations for"), top_k: int = Query(6, description="Number of recommendations")):
    """Get movie recommendations based on input movie title using sentence transformer"""
    if sentence_model is None or embeddings is None:
        init_sentence_transformer()
        if sentence_model is None:
            raise HTTPException(status_code=500, detail="Sentence transformer model not initialized.")
    
    df, _ = load_model_data()
    
    # Preprocess the movie title: remove special characters and lowercase
    movie_clean = re.sub(r'[^\w\s]', '', movie).lower()
    
    # Find closest matching title in dataframe for better user experience
    all_titles = df['Title'].str.lower().tolist()
    closest_match = get_close_matches(movie_clean, all_titles, n=1, cutoff=0.3)
    
    if closest_match:
        movie_title = closest_match[0]
        movie_index = df[df['Title'].str.lower() == movie_title].index[0]
        original_title = df.iloc[movie_index].Title
    else:
        # If no match found, use the input directly
        original_title = movie
    
    # Encode the input movie
    query_embedding = sentence_model.encode(movie_clean, convert_to_tensor=True)
    
    # Compute cosine similarity with all precomputed embeddings
    cosine_scores = util.cos_sim(query_embedding, embeddings)[0]
    
    # Get top-k most similar titles
    top_results = torch.topk(cosine_scores, k=top_k + 5)  # Get a few extra to skip potential self-matches
    
    # Format recommendations
    recommendations = []
    count = 0
    
    for score, idx in zip(top_results[0], top_results[1]):
        title = df.iloc[idx.item()].Title
        if title.lower() != movie_clean and count < top_k:  # Skip exact self-match
            recommendations.append(MovieRecommendation(
                title=title,
                movie_id=int(df.iloc[idx.item()].movie_id),
                similarity=float(score)
            ))
            count += 1
        if count >= top_k:
            break
    
    return RecommendationResponse(
        recommendations=recommendations,
        query=movie,
        matched_query=original_title
    )

@app.get("/movies", response_model=List[str])
def get_movies(limit: int = Query(100, description="Number of movies to return")):
    """Get list of movies available in the dataset"""
    df, _ = load_model_data()
    return df['Title'].tolist()[:limit]

# Startup event
@app.on_event("startup")
async def startup_event():
    """Check if model files exist on startup and initialize models"""
    try:
        # Load traditional similarity model
        load_model_data()
        print("Traditional similarity model loaded successfully!")
        
        # Initialize sentence transformer
        init_sentence_transformer()
    except Exception as e:
        print(f"Warning: {e}")
        print("You need to train and save the model first.")

# Run the app
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)