import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from surprise import KNNBasic, KNNWithMeans, KNNWithZScore, KNNBaseline, Dataset, Reader
import re
from collections import Counter

# Caching the data load function
@st.cache_data
def load_data():
    url1 = 'https://raw.githubusercontent.com/Ao-Peng/Streamlit-App/main/movies.csv'
    url2 = 'https://raw.githubusercontent.com/Ao-Peng/Streamlit-App/main/ratings.csv'
    movies_data = pd.read_csv(url1)
    ratings_data = pd.read_csv(url2)
    
    # Extract year from movie titles and create a new column
    movies_data['year'] = movies_data['title'].str.extract('\((\d{4})\)')
    movies_data['year'] = pd.to_numeric(movies_data['year'], errors='coerce')
    
    # Split genres into lists for easier filtering
    movies_data['genre_list'] = movies_data['genres'].str.split('|')
    return movies_data, ratings_data


# Function to gather similar users' data for a movie
def get_similar_users_data(movie_id, algo, new_user_id, trainset):
    sim_users = algo.get_neighbors(trainset.to_inner_uid(new_user_id), k=min(200, trainset.n_users))

    similar_users_ratings = {}
    for user_id in sim_users:
        user_ratings = ratings_data[(ratings_data['userId'] == trainset.to_raw_uid(user_id)) & 
                                    (ratings_data['movieId'] == movie_id)]['rating']
        if not user_ratings.empty:
            similar_users_ratings[trainset.to_raw_uid(user_id)] = user_ratings.values[0]

    return similar_users_ratings

# Function to plot distribution of similar users' ratings
def plot_ratings_distribution(similar_users_ratings):
    fig, ax = plt.subplots(figsize=(6, 3))
    sns.histplot(similar_users_ratings, bins=10, ax=ax)
    ax.set_title('Distribution of Ratings from Similar Users')
    ax.set_xlabel('Rating')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)
    
    
# Function to count high ratings given by similar users after the year threshold
def count_high_ratings_after_year(similar_users_data, year_threshold):
    high_ratings_after_year = 0
    for user, rating in similar_users_data.items():
        rating_year = ratings_data[(ratings_data['userId'] == user) & 
                                   (ratings_data['rating'] >= 4)]['timestamp'].max()
        if rating_year and rating_year > year_threshold:
            high_ratings_after_year += 1
    return high_ratings_after_year

# Function to find the most prominent period for high ratings
def find_prominent_period(similar_users_data):
    rating_years = []
    for user, rating in similar_users_data.items():
        rating_year = ratings_data[(ratings_data['userId'] == user) & 
                                   (ratings_data['rating'] >= 4)]['timestamp'].max()
        if rating_year:
            rating_years.append(rating_year)
    if rating_years:
        period_start = min(rating_years)
        period_end = max(rating_years)
        return period_start, period_end
    else:
        return None, None
    
    
movies_data, ratings_data = load_data()

# Surprise dataset and reader
reader = Reader(rating_scale=(0, 5))
data = Dataset.load_from_df(ratings_data[['userId', 'movieId', 'rating']], reader)

# Algorithm list
algorithms = [
    ("KNNBasic", KNNBasic()),
    ("KNNWithMeans", KNNWithMeans()),
    ("KNNWithZScore", KNNWithZScore()),
    ("KNNBaseline", KNNBaseline()),
    #("SVD", SVD()),
    #("SVDpp", SVDpp()),
    #("NMF", NMF()),
    #("SlopeOne", SlopeOne()),
    #("CoClustering", CoClustering())
]

# Streamlit UI components
st.title("Movie Recommendation App🎬")

# Genre filter for UI
unique_genres = sorted(set(sum(movies_data['genre_list'].tolist(), [])))
genre_filter = st.multiselect('Filter movies by genre:', options=unique_genres)

# Year filter for UI
min_year, max_year = int(movies_data['year'].min()), int(movies_data['year'].max())
year_filter = st.slider('Filter movies by release year:', min_value=min_year, max_value=max_year, value=(min_year, max_year))

# Filtered data for UI based on selections
filtered_movies = movies_data.copy()
if genre_filter:
    filtered_movies = filtered_movies[filtered_movies['genre_list'].apply(lambda x: any(genre in x for genre in genre_filter))]
if year_filter:
    filtered_movies = filtered_movies[(filtered_movies['year'] >= year_filter[0]) & (filtered_movies['year'] <= year_filter[1])]


# Select movies
selected_movies = st.multiselect("Select movies that you've seen and found interesting😻", filtered_movies['title'].unique())

# Number of recommendations
num_recommendations = st.slider("Select the number of recommendations", min_value=1, max_value=10, value=5)

# Submit button
submit_button = st.button("Submit")

if submit_button and selected_movies:
    try:
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

        # Train all algorithms
        trainset = updated_data.build_full_trainset()
        # Initialize movie scores
        movie_scores = Counter()

        # Train all algorithms and get top N movies from movies rated by top 200 similar users
        trainset = updated_data.build_full_trainset()
        for algo_name, algo in algorithms:
            algo.fit(trainset)

            # Get top 200 similar users for the new user
            sim_users = algo.get_neighbors(trainset.to_inner_uid(new_user_id), k=min(200, trainset.n_users))

            # Get movies rated by the top 50 similar users
            similar_users_movies = set()
            for user_id in sim_users:
                similar_users_movies.update(ratings_data[ratings_data['userId'] == trainset.to_raw_uid(user_id)]['movieId'])

            # Calculate ratings for these movies using the algorithm
            predictions = algo.test([(new_user_id, movie_id, 0) for movie_id in similar_users_movies if movie_id not in new_user_df['movieId'].values])


            # Sort predictions by estimated rating
            sorted_predictions = sorted(predictions, key=lambda x: x.est, reverse=True)

            # Get total number of users
            total_users = ratings_data['userId'].nunique()

            # Update movie scores based on their rank in the top N recommendations, 
            # the number of similar users who rated it, 
            # the percentage of similar users who gave a high rating, and the total number of users
            for rank, pred in enumerate(sorted_predictions[:num_recommendations], start=1):
                # Get similar users' data for the movie
                similar_users_data = get_similar_users_data(pred.iid, algo, new_user_id, trainset)
                num_similar_users = len(similar_users_data)
                num_high_ratings = len([rating for rating in similar_users_data.values() if rating >= 4])
                high_rating_percentage = 5*num_high_ratings / num_similar_users if num_similar_users > 0 else 0
                
                # Normalize the number of similar users who rated the movie by dividing it by the total number of users
                normalized_num_similar_users = num_similar_users / total_users
                
                # Adjust the weight given to the high rating percentage by multiplying it by a factor
                adjusted_high_rating_percentage = high_rating_percentage * 0.5
                movie_scores[pred.iid] += (num_recommendations + 1 - rank) * normalized_num_similar_users * adjusted_high_rating_percentage



        # Sort movies based on their scores
        sorted_movies = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)

        # Display recommendations
        st.write("## Recommendations:")
        for i, (movie_id, score) in enumerate(sorted_movies[:num_recommendations], start=1):
            movie_title = movies_data[movies_data['movieId'] == movie_id]['title'].values
            if len(movie_title) > 0:
                # Get similar users' data for the movie
                similar_users_data = get_similar_users_data(movie_id, algo, new_user_id, trainset)
                num_similar_users = len(similar_users_data)
                num_high_ratings = len([rating for rating in similar_users_data.values() if rating >= 4])
                st.write(f"{i}. {movie_title[0]} - Score: {score}")
                st.write(f"{num_similar_users} similar users, who share similar tastes and {num_high_ratings} of them also gave a high rating to this movie.")

                # Feature Contribution Analysis: Explain the influence of genres and user ratings
                st.write(f"Genres: {', '.join(movies_data[movies_data['movieId'] == movie_id]['genre_list'].values[0])}")

                # Get similar users' ratings for the recommended movie
                similar_users_ratings = get_similar_users_data(movie_id, algo, new_user_id, trainset)

                # Plot the distribution of similar users' ratings
                plot_ratings_distribution(similar_users_ratings)
                
#                 # Get the most prominent period for high ratings
#                 period_start, period_end = find_prominent_period(similar_users_data)

#                 # Set the year threshold based on the identified period
#                 if period_end:
#                     year_threshold = period_end.year - 10  # Set the threshold 10 years before the end of the prominent period
#                 else:
#                     year_threshold = None

#                 # Count high ratings given by similar users after the year threshold
#                 if year_threshold:
#                     high_ratings_after_year = count_high_ratings_after_year(similar_users_data, year_threshold)
#                 else:
#                     high_ratings_after_year = 0

#                 # Write the explanation
#                 if period_start and period_end:
#                     st.write(f"Rated by peers: {high_ratings_after_year} users gave a high rating after {period_start.year}")
#                     st.write(f"Most prominent period for high ratings: {period_start.year} - {period_end.year}")
#                 else:
#                     st.write("No high ratings found for the movie among similar users.")
                # Confidence Scores: Display prediction uncertainty
                st.write(f"Confidence score: {pred.est:.2f}")

        pass
    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.warning("Please select at least one movie and apply filters before submitting.")