# Movie Recommendation API

This project is a FastAPI-based backend service for recommending movies based on user input. It offers two recommendation strategies: one using traditional cosine similarity on metadata, and another using a sentence transformer model to find semantically similar movies.

Whether you're building a movie discovery app or experimenting with recommendation systems, this API is designed to be easy to use, flexible, and fast.

---

## Resources

- [Google Colab Notebook](https://colab.research.google.com/drive/1O8ggT1vwPhG7Uzk_H2fbDvrweO52ANVv?usp=sharing) - Google colab link
- [Model pkl file ](https://drive.google.com/file/d/1V7tuTITsVt79f1iSBdyiB5nUOKGCFTj7/view?usp=sharing) - Sentence transformer model weights
- [Similarity_matrix pkl file ](https://drive.google.com/file/d/1ImZNT1MRE5H09VqFhLbQs-c-mEOpyoZp/view?usp=sharing) - Similarity matrix file

---

## Features

- Traditional movie recommendations using a precomputed similarity matrix.
- Semantic recommendations powered by Sentence Transformers (`all-MiniLM-L6-v2`).
- Fuzzy matching to handle minor typos or variations in movie titles.
- Ready-to-use REST API with automatic documentation (Swagger).
- CORS support for frontend integration.

---

## Requirements

- Python 3.8 or higher

### Installing dependencies

```bash
pip install -r requirements.txt
```
