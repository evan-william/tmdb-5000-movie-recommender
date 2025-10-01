Movie Recommendation EngineA content-based movie recommender built with a clean, modern web interface using Python, Flask, and Scikit-learn.OverviewThis project implements a content-based filtering model to solve a classic recommendation problem. Originally a data science script, it has been transformed into an interactive web application where users can enter a movie they like and instantly get a list of 5 similar movies.It uses TF-IDF Vectorization and Cosine Similarity, powerful techniques for measuring the similarity between items based on their textual content. The model learns the "profile" of a movie from its genre, keywords, cast, and crew to make its recommendations.What I LearnedText Processing & Feature Engineering: Parsing and cleaning complex text data (genres, cast, keywords) to create a meaningful "content soup" for each movie.Text Vectorization: Converting text profiles into numerical vectors using TfidfVectorizer.Cosine Similarity: Implementing a mathematical measure to find the "distance" between movies in a high-dimensional space.Scikit-learn for ML: Applying core machine learning concepts for an unsupervised recommendation task.Web Development with Flask: Creating routes, handling form submissions, and rendering dynamic templates.Full-Stack Connection: Wiring a Python machine learning backend to a web frontend to create a complete application.Technical StackPython: Core programming languageFlask: Web framework for the backendScikit-learn: For vectorization and similarity calculationPandas: For data manipulationHTML & CSS: For the frontend user interfaceKey Concepts AppliedThe core of the recommender is the transformation of movie metadata into a numerical matrix, which is then used to calculate similarity scores. The entire data processing and model building pipeline runs once when the application starts.# Create a "tags" column representing the movie's content
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

# Vectorize the text and compute the similarity matrix
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
vectors = vectorizer.fit_transform(new_df['tags']).toarray()
similarity = cosine_similarity(vectors)
How It WorksModel Building: When the Flask application starts, it loads the tmdb_5000_movies.csv and tmdb_5000_credits.csv datasets, processes them, and builds the cosine similarity matrix just once.User Input: A user visits the web page and submits a movie title through an HTML form.Backend Recommendation: The Flask backend receives the title, finds the movie in the pre-computed similarity matrix, and retrieves the top 5 most similar movies.Display Results: The application re-renders the webpage, dynamically displaying the list of recommended movies.Installation & Setup# Clone the repository
git clone [https://github.com/username/movie-recommender.git](https://github.com/username/movie-recommender.git)
cd movie-recommender

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt

# Place the dataset files in the root directory
# - tmdb_5000_movies.csv
# - tmdb_5000_credits.csv

# Run the application
python app.py
RequirementsYou can install all dependencies from the requirements.txt file.requirements.txtFlask
scikit-learn
pandas
numpy
Python 3.8 or higher is recommended.Project StructureThe project is organized to separate the Flask application from the machine learning logic.movie-recommender/
├── app.py                      # Main Flask application file
├── recommender.py              # The recommender class and ML logic
├── requirements.txt            # Project dependencies
├── tmdb_5000_movies.csv        # The movies dataset
├── tmdb_5000_credits.csv       # The credits dataset
├── static/
│   └── style.css               # CSS for the user interface
└── templates/
    └── index.html              # The HTML file for the user interface
Things I'd ImproveAdd Movie Posters: Fetch movie poster images from an API (like TMDB's own API) to make the UI more visual.Explore Other Models: Implement a collaborative filtering model (e.g., using SVD) if user ratings were available.Cache the Model: Save the processed data and similarity matrix to a file (e.g., using pickle) to avoid re-calculating on every app startup.Containerize: Package the application with Docker for easier deployment and scalability.Deploy to the Cloud: Host the application on a service like Heroku or AWS so anyone can access it.Author[Your Name] Version 1.0 (2025)This project was a fantastic exercise in building a complete machine learning application from end to end. It showcases how data processing, machine learning modeling, and web development can be combined to create a practical, interactive tool.If you have any feedback or suggestions, feel free to open an issue or pull request!This project is for educational purposes. Feel free to fork, modify, or use it for your own learning.