import pandas as pd
import numpy as np
import sys
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from surprise import Reader, Dataset, SVD, model_selection
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import warnings; warnings.simplefilter('ignore')



userId = pd.read_csv('userId.csv')

features = ['age','gender','occupation']

def combined_feature(row):
    return str(row['age'])+" "+row['gender']+" "+row['occupation']+" "+str(row['zip code'])

userId['combined_feature'] = userId.apply(combined_feature, axis=1)

count_n = TfidfVectorizer()
count_matrix_n = count_n.fit_transform(userId['combined_feature'])
cosine_sim = linear_kernel(count_matrix_n)





    

user_id = int(sys.argv[1])

userId = userId.reset_index()
cluster = userId['userId']
indices = pd.Series(userId.index, index=userId['userId'])
def get_cluster(id):
    idx = indices[id]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sortedUserList = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    topTen = sortedUserList[0:10]
    user_group = [i[0] for i in topTen]
    return user_group
user = []


user_cluster = get_cluster(user_id)

for val in range(len(user_cluster)):
    if (user_cluster[val] + 1 != user_id):
        user.append(user_cluster[val] +1)

random_select_user_from_cluster = []
rating = pd.read_csv('ratings_small.csv')['userId']
rating = rating.values.tolist()

def select_random_user_from_cluster_function(userId):
    count = 0
    for val in range(len(rating)):
        if rating[val] == userId:
            count = count + 1
        if (count == 50):
            random_select_user_from_cluster.append(userId)
            break

for id in range(len(user)):
    select_random_user_from_cluster_function(user[id])

print("This is the similar user list ",random_select_user_from_cluster)




#collaborative filtering
reader = Reader()
ratings = pd.read_csv('ratings_small.csv')

ratings = ratings.drop(columns='timestamp')
data = Dataset.load_from_df(ratings, reader)
svd = SVD()
model_selection.cross_validate(svd, data, measures=['RMSE', 'MAE'], cv=3)

trainset = data.build_full_trainset()
svd.fit(trainset)




# recoomand movie to the user according to the selected group

rating = pd.read_csv('ratings_small.csv')

movies = pd.read_csv('movies.csv')

rating_t = rating.groupby('userId').filter(lambda x: len(x) >= 50)



rating = pd.merge(movies, rating_t)
user_rating = rating.pivot_table(index=['userId'], columns=['title'], values='rating')
user_rating = user_rating.fillna(0)
item_similarity = user_rating.corr(method='pearson')




def get_user_cluster_movie(movie_name, rating):
    similar_score = item_similarity[movie_name] * (rating - 2.5)
    similar_score = similar_score.sort_values(ascending=False)
    return similar_score

movie_list = []

rating_list=rating.values.tolist()
for val in range(len(random_select_user_from_cluster)):
    for j in range(len(rating_list)):
        if random_select_user_from_cluster[val] == rating_list[j][3]:
            movie_list.append((rating_list[j][1],rating_list[j][4]))

similar_movies = pd.DataFrame()
for movie, rating in movie_list:
    similar_movies = similar_movies.append(get_user_cluster_movie(movie, rating))


similar_movies = similar_movies.sum().sort_values(ascending=False)
similar_movies = similar_movies.head(10)
similar_moviesid_list = []

movieList = movies.values.tolist()
movieTitle = similar_movies.index.tolist()
for movie in range(len(movieTitle)):
    for list in range(len(movieList)):
        if movieTitle[movie] == movieList[list][1]:
            similar_moviesid_list.append(movieList[list][0])

def getEst(userId,movieId):
    return svd.predict(userId, movieId).est

print(similar_movies)

for val in range(len(similar_moviesid_list)):
    print("Predict movie rating for the input user",getEst(int(user_id),similar_moviesid_list[val]))




