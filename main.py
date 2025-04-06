from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import difflib
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load and process movie data
movies = pd.read_csv('movies.csv')
movies.reset_index(inplace=True)  # Add index column for referencing like the notebook

# Fill missing values
selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']
for feature in selected_features:
    movies[feature] = movies[feature].fillna('')

# Combine selected features into a single string
combined_features = (
    movies['genres'] + ' ' +
    movies['keywords'] + ' ' +
    movies['tagline'] + ' ' +
    movies['cast'] + ' ' +
    movies['director']
)

# Vectorize features and calculate similarity
vectorizer = TfidfVectorizer()
feature_vector = vectorizer.fit_transform(combined_features)
similarity = cosine_similarity(feature_vector)

# TMDb API Key
TMDB_API_KEY = "a0082fc71a70cb7cb32282e23d1d69f5"

# Fetch poster using TMDb
def fetch_poster(title):
    url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={title}"
    response = requests.get(url).json()
    if response.get("results"):
        poster_path = response["results"][0].get("poster_path")
        if poster_path:
            return f"https://image.tmdb.org/t/p/w500{poster_path}"
    return "https://via.placeholder.com/300"

# Recommendation logic
def recommend(movie_name):
    # Try to find exact match (case-insensitive)
    exact_matches = movies[movies['title'].str.lower() == movie_name.lower()]

    recommended_titles = []
    poster_urls = []

    if not exact_matches.empty:
        exact_movie_index = exact_matches['index'].values[0]
        exact_title = exact_matches['title'].values[0]
        recommended_titles.append(exact_title)
        poster_urls.append(fetch_poster(exact_title))
    else:
        # Use close match if exact not found
        movie_list = movies['title'].tolist()
        close_matches = difflib.get_close_matches(movie_name, movie_list, n=1)
        if not close_matches:
            return [], []

        exact_movie_index = movies[movies['title'] == close_matches[0]]['index'].values[0]
        exact_title = close_matches[0]
        recommended_titles.append(exact_title)
        poster_urls.append(fetch_poster(exact_title))

    # Now get similar movies (excluding the one we already added)
    similarity_scores = list(enumerate(similarity[exact_movie_index]))
    sorted_similar_movies = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    count = 0
    for i, _ in sorted_similar_movies:
        title = movies.iloc[i]['title']
        if title != exact_title:
            recommended_titles.append(title)
            poster_urls.append(fetch_poster(title))
            count += 1
        if count == 5:
            break

    return recommended_titles, poster_urls

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend_route():
    movie_name = request.form['movie']
    recommended_movies, posters = recommend(movie_name)
    movie_data = list(zip(recommended_movies, posters))
    return render_template('reco.html', movie_data=movie_data)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
