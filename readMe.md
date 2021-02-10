# In this project we are comparing two approaches for movie recommendation for a new user or existing user based on their age,gender,occupation

We have implemented all the method from scratch and tried to improve the existing movie recommender system.

# First install all the imports : you can use pip to install all these imports

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

# Approach 1: run- python Approch1_movieRecommendation.py 943

1. For this you need to run the movieRecommendation.py from the command line and give user_id from 1 to 943
2. Command - python Approch1_movieRecommendation.py 943 (you can can value from 1 to 943)

# Code Implementation Steps

1. First we combined the features(age,gender,occupation)
2. We pass this combined feature to TfidfVectorizer to get the cosine similarity between all the user
3. Now we group the user according to the cosine similarities

# Result: This is the similar user list [442, 501, 245, 304, 327, 361]

4. Next we take these users make the tuple list of the movies and ratings from the rating.csv and movies.csv
5. Then we recommend movies based on the total score and predict the user rating for each recommended movie

# Final Result: For user 943
Movies Score
Reservoir Dogs (1992) 228.609784
Goodfellas (1990) 221.945500
Jackie Brown (1997) 219.388194
O Brother, Where Art Thou? (2000) 218.369608
Ocean's Eleven (2001) 217.387415
Jerry Maguire (1996) 215.236588
Insomnia (2002) 215.113956
Untouchables, The (1987) 214.845558
Groundhog Day (1993) 214.387803
Big (1988) 210.148683



predict movie rating for the input user 4.150698478010207
predict movie rating for the input user 4.117589700825852
predict movie rating for the input user 3.726632367400274
predict movie rating for the input user 3.9453673537067337
predict movie rating for the input user 3.959799932586692
predict movie rating for the input user 3.764032989170141
predict movie rating for the input user 3.5648023445428016
predict movie rating for the input user 3.8859466699988907
predict movie rating for the input user 3.8388138667957135
predict movie rating for the input user 3.8606186256032706

# Approch 2: run - python Approch2_improvedMovieRecommendation.py 943

1. For this you need to run the movieRecommendation.py from the command line and give user_id from 1 to 943
2. Command - python Approch2_improvedMovieRecommendation.py 943 (you can can value from 1 to 943)

# Code Implementation Steps

1. First we combined the features(age,gender,occupation)
2. We pass this combined feature to TfidfVectorizer to get the cosine similarity between all the user
3. Now we make the undirected graph of all the similar user and here we make cluster of more than 5 users group using their combined feature

# User clusters from undirected graph
user_group_list combined_feature
0 943 22 M student # this is the input user
1 0|3|455|888|716|831 24 M technician
2 736|16|675|474|794 30 M programmer
3 32|65|705|36|837|134|390|48|407|476|158 23 M student
4 481|641|452|587|269|367|527|591|786|51|630|631... 18 F student
5 624|513|113|159|644|57|863 27 M programmer
6 256|66|581|645|903|618|396|620|366|340|760|699... 17 M student
7 434|626|68|104|267 24 M engineer
8 640|516|72|874|300|368|471|347 24 M student
9 704|258|323|322|227|197|495|80|275|724|922|541... 21 F student
10 240|770|659|756|102|938|93 26 F student
11 848|100|280|617|460 15 F student
12 912|483|757|103|285|428|653 27 M student
13 788|923|108|477|511 29 M other
14 546|677|136|908|589|877|622|157 50 M educator
15 248|354|583|202|306|371|726|247|152|153|892 25 M student
16 402|580|166|668|701|718 37 M other
17 224|787|437|327|890 51 F administrator
18 594|550|284|621|799 25 M programmer
19 688|341|918|696|906|733|398 25 M other
20 416|804|436|904|504 27 F other
21 913|732|698|459|796|543 44 F other

4. Next, we take the cosine similarity for the input user with the Users clusters from undirected graph

# Result: This is the similar user list [442, 501, 245, 304, 327, 361]

5. Next we take these users make the tuple list of the movies and ratings from the rating.csv and movies.csv
6. Then we recommend movies based on the total score and predict the user rating for each recommended movie

## movies,scores
Fisher King, The (1991) 755.188866
Unforgiven (1992) 744.529194
Fast Times at Ridgemont High (1982) 730.106762
Misery (1990) 720.259366
Three Kings (1999) 718.417139
Jaws (1975) 713.815785
Jackie Brown (1997) 711.431124
Ghostbusters II (1989) 709.595190
Insomnia (2002) 706.948516
Untouchables, The (1987) 688.182905


Predict movie rating for the input user 3.6438764529624676
Predict movie rating for the input user 3.866623697902947
Predict movie rating for the input user 3.6783744809250707
Predict movie rating for the input user 3.9056005710274846
Predict movie rating for the input user 3.796831264353779
Predict movie rating for the input user 3.851507019542307
Predict movie rating for the input user 3.7547849040923857
Predict movie rating for the input user 3.025542714844357
Predict movie rating for the input user 3.583938185472557
Predict movie rating for the input user 3.9141295113729297
