from flask import Flask, render_template, request
from recommender import load_and_process_data, get_recommendations
import sys

app = Flask(__name__)

# load data & models
print("Initializing the recommendation engine...")
df, similarity_matrix = load_and_process_data()

# exit if loading fails...
if df is None or similarity_matrix is None:
    print("Failed to initialize. Exiting application.")
    sys.exit(1)

# get list for movie drop down
movie_titles = df['title'].tolist()

@app.route('/', methods=['GET', 'POST'])
def home():
    recommendations = None
    selected_movie = ""
    if request.method == 'POST':
        selected_movie = request.form.get('movie')
        if selected_movie:
            recommendations = get_recommendations(selected_movie, df, similarity_matrix)
    
    return render_template(
        'index.html', 
        movie_titles=movie_titles, 
        recommendations=recommendations,
        selected_movie=selected_movie
    )

if __name__ == '__main__':
    app.run(debug=True)
