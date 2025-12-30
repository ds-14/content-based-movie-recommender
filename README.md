# Content-Based Movie Recommendation System

## Overview
This project implements a content-based movie recommendation system using natural language processing and machine learning techniques. It recommends movies similar to a selected title based on textual features extracted from movie metadata.

## Objective
To build a personalized movie recommender using content similarity rather than user ratings, and deploy it as an interactive web application.

## Dataset
The project uses movie metadata derived from the TMDB dataset, including textual attributes such as movie titles and descriptions.

> Due to size and licensing constraints, the dataset and trained model files are not included in this repository.

## Methodology
- Text preprocessing and TF-IDF vectorization
- Similarity computation using sigmoid kernel
- Content-based filtering for recommendations
- Model serialization using Joblib
- Interactive web app development using Streamlit

## Features
- Recommends top 10 similar movies
- Content-based filtering (no user history required)
- Simple and intuitive Streamlit interface

## Tools & Technologies
- Python
- Pandas, NumPy
- Scikit-learn
- NLP (TF-IDF Vectorization)
- Streamlit

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/content-based-movie-recommender.git
2. Navigate to the project directory:
```bash
   cd content-based-movie-recommender
```

3. Install dependencies:
```bash
   pip install -r requirements.txt
```

4. Run the Streamlit application:
```
  streamlit run movie_recommendation_app.py
