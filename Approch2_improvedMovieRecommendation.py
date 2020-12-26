import pandas as pd
import numpy as np
import sys
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from surprise import Reader, Dataset, SVD, model_selection
from sklearn.metrics import mean_squared_error
from itertools import combinations
import networkx as nx

import warnings; warnings.simplefilter('ignore')



userId = pd.read_csv('userId.csv')
features = ['age','gender','occupation']

def combined_feature(row):
    return str(row['age'])+" "+row['gender']+" "+row['occupation']

userId['combined_feature'] = userId.apply(combined_feature, axis=1)


count_n = TfidfVectorizer()
count_matrix_n = count_n.fit_transform(userId['combined_feature'])

cosine_sim = linear_kernel(count_matrix_n)


user_id =int(sys.argv[1])
userId = userId.reset_index()
indices_info = pd.Series(userId['combined_feature'], index=userId.index)

cosine_simi_vector = []
storeNodes =[]
for i in range(len(cosine_sim)):
    for j in range(len(cosine_sim[i])):
        if cosine_sim[i][j] >= 1 and i != j:
            storeNodes.append(i)
            storeNodes.append(j)
            cosine_simi_vector.append((i, j))


storeNodes = list(set(storeNodes))
cosine_simi_vector = list(cosine_simi_vector)



from networkx.algorithms.community import k_clique_communities
graph = nx.Graph()
graph.add_nodes_from(range(len(storeNodes)))

for node1, node2 in cosine_simi_vector:
    graph.add_edge(node1, node2)
c = list(k_clique_communities(graph,5))

convert_forzenset=c 

user_group_list = [list(x) for x in convert_forzenset]

userId = userId.reset_index()
indices = pd.Series(userId['combined_feature'] , index=userId.index)

get_combined_feature = [indices_info[user_id-1]]

for x in range(len(user_group_list)):
    get_combined_feature.append(indices[user_group_list[x][0]])




user_group_list_concat = [user_id]
for i in range(len(user_group_list)):
    user_group_list_concat.append('|'.join(str(x) for x in user_group_list[i]))

cluster_user_dataframe = pd.DataFrame(
    {'user_group_list': user_group_list_concat,
     'combined_feature': get_combined_feature
    })
print("User clusters from undirected graph")
print(cluster_user_dataframe)


count_n_cluster = CountVectorizer()
count_matrix_n_cluster = count_n.fit_transform(cluster_user_dataframe['combined_feature'])
cosine_sim_cluster = cosine_similarity(count_matrix_n_cluster)

def get_cluster():
    sim_scores = list(enumerate(cosine_sim_cluster[0]))
    sortedUserList = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    topTen = sortedUserList[1:3]
    user_group = [i[0] for i in topTen]
    return user_group

main_cluster = get_cluster()
group_user_str=''
for i in range(len(main_cluster)):
    group_user_str += cluster_user_dataframe.iloc[[main_cluster[i]]]['user_group_list'].tolist()[0] + '|'
    
group_user_str = group_user_str.split("|")
group_user_str = group_user_str[:len(group_user_str) - 1]


user_cluster = [int(x) for x in group_user_str]

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

for id in range(len(user_cluster)):
    if user_cluster[id]+1 != user_id:
        select_random_user_from_cluster_function(user_cluster[id]+1)

print("Better User Cluster group",random_select_user_from_cluster)



# same as before

reader = Reader()
ratings = pd.read_csv('ratings_small.csv')

ratings = ratings.drop(columns='timestamp')
data = Dataset.load_from_df(ratings, reader)
svd = SVD()

model_selection.cross_validate(svd, data, measures=['RMSE', 'MAE'],cv=3)

trainset = data.build_full_trainset()
svd.fit(trainset)

# recommand movie to the user according to the selected group

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




