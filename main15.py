import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# Wczytanie danych
movies = pd.read_csv('movies.csv')

# Przetworzenie danych
cv = CountVectorizer()
count_matrix = cv.fit_transform(movies['description'].fillna(''))

# Obliczenie podobieństwa kosinusowego
cosine_sim = cosine_similarity(count_matrix, count_matrix)

# Funkcja zwracająca rekomendacje na podstawie podobieństwa kosinusowego
def get_recommendations(title):
    idx = movies[movies['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices]

# Przykład użycia
recommendations = get_recommendations('The Dark Knight')
print(recommendations)
