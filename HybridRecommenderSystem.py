import pandas as pd

"""Step 1: read movie and rating datasets"""

movie = pd.read_csv("movie.csv")
rating = pd.read_csv("rating.csv")

"""Step 2: add the movie titles and genres corresponding to the Ids to 
the rating dataset from the movie dataset"""

df = movie.merge(rating, how="left", on="movieId")
df.head()
df.shape

"""Step 3 : Keep the names of movies with a total vote count below 1000 in a list and remove them from the dataset """
comment_counts = pd.DataFrame(df["title"].value_counts())
rare_movies = comment_counts[comment_counts["count"] < 1000].index
common_movies = df[~df["title"].isin(rare_movies)]
common_movies.shape


"""Step 4 : Create a pivot table for the dataframe where userID's are in the index, film names
 are in the columns, and ratings are present as values"""


user_movie_df = common_movies.pivot_table(index= ["userId"], columns=["title"], values="rating")


"""Step 5: Functionization  """

def create_user_movie_df():
    import pandas as pd
    movie = pd.read_csv("movie.csv")
    rating = pd.read_csv("rating.csv")
    df = movie.merge(rating, how="left", on="movieId")
    comment_counts = pd.DataFrame(df["title"].value_counts())
    rare_movies = comment_counts[comment_counts["count"] <= 1000].index
    common_movies = df[~df["title"].isin(rare_movies)]
    user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values= "rating")
    return user_movie_df

user_movie_df = create_user_movie_df()

"""Step 6: Select a random user ID """

random_user = 1997

"""Step 7: "Create a new dataframe named random_user_df consisting of observation units belonging to the selected user."""

random_user_df = user_movie_df[user_movie_df.index == random_user]

random_user_df.head()

"""Step 8: Assign the movies that the selected user has voted for to a list named movies_watched."""

movies_watched = random_user_df.columns[random_user_df.notna().any()].tolist()

"""Step 9: Select the columns related to the movies watched by the selected user from 
user_movie_df and create a new dataframe named movies_watched_df"""

movies_watched_df = user_movie_df[movies_watched]
movies_watched_df.head()
movies_watched_df.shape

"""Step 10 : Create a new dataframe named user_movie_count that contains information about
 how many of the selected user's watched movies each user has watched."""

user_movie_count = movies_watched_df.T.notnull().sum()
user_movie_count = user_movie_count.reset_index()
user_movie_count.columns = ["userId", "movie_count"]

"""Step 11: Create a list named users_same_movies consisting of user IDs from those who have 
watched 60% or more of the movies that the selected user has voted for."""

perc = len(movies_watched)* 0.6
user_same_movies = user_movie_count[user_movie_count["movie_count"] > perc]["userId"]
len(user_same_movies)

"""Step 12: Filter the dataframe movies_watched_df to include only the user IDs of users in the
 users_same_movies list who show similarity with the selected user."""

final_df = movies_watched_df[movies_watched_df.index.isin(user_same_movies)]
final_df.head()
final_df.shape


"""Step 13 : Create a new dataframe named corr_df that will contain the correlations between users"""

corr_df = final_df.T.corr().unstack().sort_values()
corr_df = pd.DataFrame(corr_df, columns=["corr"])
corr_df.index.names = ["user_id_1", "user_id_2"]
corr_df = corr_df.reset_index()
corr_df[corr_df["user_id_1"]==random_user]


"""Step 14: Filter the users with high correlation (above 0.65) with the selected user and 
create a new dataframe named top_users"""

top_users = corr_df[(corr_df["user_id_1"] == random_user) & (corr_df["corr"] >= 0.65)][["user_id_2", "corr"]].reset_index(drop=True)
top_users = top_users.sort_values(by='corr', ascending=False)
top_users.rename(columns={"user_id_2": "userId"}, inplace=True)
top_users.shape
top_users.head()


"""Step 15:Merge the rating dataset with the top_users dataframe """


top_users_ratings = top_users.merge(rating[["userId", "movieId", "rating"]], how='inner')
top_users_ratings = top_users_ratings[top_users_ratings["userId"] != random_user]
top_users_ratings.head()


"""Step 16: Create a new variable named weighted_rating, which is the product of each user's 
corr and rating values """

top_users_ratings['weighted_rating'] = top_users_ratings['corr'] * top_users_ratings['rating']
top_users_ratings.head()

""" Step 17: Create a new dataframe named recommendation_df that includes the average value of
 weighted ratings for each movie ID and all users associated with that movie"""

recommendation_df = top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"})
recommendation_df = recommendation_df.reset_index()
recommendation_df.head()
""" Step 18 : Select the movies from recommendation_df where the weighted rating is greater than 3.5, and sort them based on the weighted rating"""

movies_to_be_recommend = recommendation_df[recommendation_df["weighted_rating"] > 3.5].sort_values("weighted_rating", ascending=False)


""" Step 19 : Retrieve the movie names from the movie dataset and select the top 5 recommended movies"""

movies_to_be_recommend.merge(movie[["movieId", "title"]])["title"][:5]