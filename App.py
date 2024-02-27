import streamlit as st
import pandas as pd
from surprise import KNNBasic, KNNWithMeans, KNNWithZScore, KNNBaseline, SVD, SVDpp, NMF, SlopeOne, CoClustering, Dataset, Reader
from surprise.prediction_algorithms.algo_base import AlgoBase
import numpy as np
from scipy.stats import pearsonr

# Load data
ratings_data = pd.read_csv('ratings.csv')
movies_data = pd.read_csv('movies.csv')

# Surprise dataset and reader
reader = Reader(rating_scale=(0.5, 5))  # Adjust rating_scale to match your data
data = Dataset.load_from_df(ratings_data[['userId', 'movieId', 'rating']], reader)

# List of algorithms from Surprise to use
algorithms = [
    ("KNNBasic", KNNBasic()),
    ("KNNWithMeans", KNNWithMeans()),
    ("KNNWithZScore", KNNWithZScore()),
    ("KNNBaseline", KNNBaseline()),
    ("SVD", SVD()),
    ("SVDpp", SVDpp()),
    ("NMF", NMF()),
    ("SlopeOne", SlopeOne()),
    ("CoClustering", CoClustering())
]

class ConstrainedPearsonR(KNNBasic):
    def __init__(self, threshold=0.0):
        KNNBasic.__init__(self, sim_options={'name': 'pearson'})
        self.threshold = threshold

    def fit(self, trainset):
        KNNBasic.fit(self, trainset)
        return self

    def estimate(self, u, i):
        ratings = [r for (_, r) in self.trainset.ir[i] if not np.isnan(r)]
        similar_users = [(user_id, self.compute_pearson_correlation(u, user_id)) for user_id in self.trainset.ir[i] if user_id != u]

        if not ratings:
            # No ratings available for the item, return a default prediction
            return self.trainset.global_mean

        weighted_sum = 0
        total_weight = 0
        for user_id, correlation in similar_users:
            if correlation > self.threshold:
                rating = self.trainset.ir[i][user_id][0] if user_id in self.trainset.ir[i] else 0
                weighted_sum += correlation * rating
                total_weight += abs(correlation)

        if total_weight > 0:
            prediction = weighted_sum / total_weight
            return prediction
        else:
            return np.mean(ratings)  # Use mean of available ratings as the prediction

    def compute_pearson_correlation(self, u, v):
        common_items = list(set(self.trainset.ur[u]) & set(self.trainset.ur[v]))
        if len(common_items) > 1:
            ratings_u = [self.trainset.ur[u][int(item)] for item in common_items]
            ratings_v = [self.trainset.ur[v][int(item)] for item in common_items]
            correlation, _ = pearsonr(ratings_u, ratings_v)
            return correlation
        else:
            return 0  # No common items, correlation is 0

algorithms.append(("ConstrainedPearsonR", ConstrainedPearsonR()))

class MeanBased(AlgoBase):
    def __init__(self):
        AlgoBase.__init__(self)

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)
        self.mean_ratings = {}
        for user, ratings in trainset.ur.items():
            self.mean_ratings[user] = np.mean([rating for (_, rating) in ratings])
        return self

    def estimate(self, u, i):
        return self.mean_ratings.get(u, self.trainset.global_mean)

algorithms.append(("MeanBased", MeanBased()))


# Streamlit app
st.title("Movie Recommendation App")

# Select movies
selected_movies = st.multiselect("Select movies you like", movies_data['title'].unique())

# Algorithm selection
algorithm_names = [algo[0] for algo in algorithms]
selected_algorithm = st.selectbox("Select algorithm", algorithm_names)

# Number of recommendations
num_recommendations = st.slider("Number of recommendations", min_value=1, max_value=10, value=5)

submit_button = st.button("Submit")

if submit_button:
    # Create a new user profile
    new_user_id = ratings_data['userId'].max() + 1
    new_user_data = {'userId': [new_user_id] * len(selected_movies),
                     'movieId': movies_data[movies_data['title'].isin(selected_movies)]['movieId'].tolist(),
                     'rating': [5] * len(selected_movies)}  # Assign a rating of 5 to all selected movies
    new_user_df = pd.DataFrame(new_user_data)

    # Append the new user data to the ratings dataset
    updated_ratings_data = pd.concat([ratings_data, new_user_df], ignore_index=True)

    # Surprise dataset and reader with the updated ratings data
    updated_data = Dataset.load_from_df(updated_ratings_data[['userId', 'movieId', 'rating']], reader)

    # Train the selected algorithm
    algorithm = dict(algorithms)[selected_algorithm]
    trainset = updated_data.build_full_trainset()
    algorithm.fit(trainset)

    # Predict ratings for the new user
    predictions = []
    for movie_id in trainset.all_items():
        predictions.append((movie_id, algorithm.predict(new_user_id, movie_id).est))

    # Sort the predicted ratings in descending order
    predictions.sort(key=lambda x: x[1], reverse=True)

    # Get top recommendations
    top_recommendations = predictions[:num_recommendations]

    # Convert movie ids to titles
    recommended_movie_titles = []
    for movie_id, _ in top_recommendations:
        movie_title = movies_data[movies_data['movieId'] == movie_id]['title'].values
        if len(movie_title) > 0:
            recommended_movie_titles.append(movie_title[0])

    # Display recommendations
    st.write("## Recommendations:")
    for i, movie_title in enumerate(recommended_movie_titles, start=1):
        st.write(f"{i}. {movie_title}")
