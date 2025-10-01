import pandas as pd
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_and_process_data(): # loads the data and return as df
    print("Loading local CSV files...")
    try:
        movies_df = pd.read_csv('tmdb_5000_movies.csv')
        credits_df = pd.read_csv('tmdb_5000_credits.csv')
    except FileNotFoundError:
        print("\nERROR: Could not find CSV files.")
        print("Please make sure 'tmdb_5000_movies.csv' and 'tmdb_5000_credits.csv' are in the root directory.\n")
        return None, None
    except Exception as e:
        print(f"\nAn error occurred while reading the CSV files: {e}\n")
        return None, None

    print("Data loaded successfully. Processing...")
    
    # merge and select feature
    movies = movies_df.merge(credits_df, on='title')
    movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
    movies.dropna(inplace=True)

    # function to help extract DATA FEATURES
    def convert(text):
        L = []
        try:
            for i in ast.literal_eval(text):
                L.append(i['name'])
        except (ValueError, SyntaxError):
            pass
        return L

    def convert_cast(text):
        L = []
        counter = 0
        try:
            for i in ast.literal_eval(text):
                if counter < 3:
                    L.append(i['name'])
                counter += 1
        except (ValueError, SyntaxError):
            pass
        return L

    def fetch_director(text):
        L = []
        try:
            for i in ast.literal_eval(text):
                if i['job'] == 'Director':
                    L.append(i['name'])
                    break
        except (ValueError, SyntaxError):
            pass
        return L
    
    def collapse(L):
        L1 = []
        for i in L:
            L1.append(i.replace(" ", ""))
        return L1

    # apply transforms
    movies['genres'] = movies['genres'].apply(convert).apply(collapse)
    movies['keywords'] = movies['keywords'].apply(convert).apply(collapse)
    movies['cast'] = movies['cast'].apply(convert_cast).apply(collapse)
    movies['crew'] = movies['crew'].apply(fetch_director).apply(collapse)
    movies['overview'] = movies['overview'].apply(lambda x: x.split())

   # create a new column "tags"
    movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
    
    new_df = movies[['movie_id', 'title', 'tags']]
    new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x).lower())

    # similiarity calculation and vectorizing THE TEXT!!!
    print("Vectorizing text and calculating similarity matrix...")
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    vectors = vectorizer.fit_transform(new_df['tags']).toarray()
    similarity = cosine_similarity(vectors)
    
    print("Processing complete.")
    return new_df, similarity

def get_recommendations(movie_title, df, similarity_matrix): # RETURNS 5 SIMILIAR MOVIE
    try:
        movie_index = df[df['title'] == movie_title].index[0]
        distances = similarity_matrix[movie_index]
        movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
        
        recommended_movies = [df.iloc[i[0]].title for i in movies_list]
        return recommended_movies
    except IndexError:
        return [f"Movie '{movie_title}' not found in the dataset."]
