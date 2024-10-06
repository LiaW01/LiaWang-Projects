# DS-GA 1004 Big Data Capstone Project
# Cecei Chen, Lia Wang, Maggie Xu


# Packages
## !pip install scikit-learn
## !pip install datasketch
import pandas as pd
import numpy as np

import cupy as cp
import cudf
from datasketch import MinHash, MinHashLSH
import heapq
from collections import defaultdict

from itertools import combinations
from scipy.stats import pearsonr

from sklearn.model_selection import train_test_split

from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder
from pyspark.sql.functions import mean, col, rand, collect_list
from pyspark.sql.types import IntegerType, DoubleType





# Problem 1:
file_path = 'full/ratings.csv'
ratings_df = cudf.read_csv(file_path)

user_movies = ratings_df.groupby('userId')['movieId'].agg(list).reset_index()
user_movies = user_movies.to_pandas().set_index('userId')['movieId'].apply(set).to_dict()


## MinHash
num_perm = 128
m_hashes = {}
lsh = MinHashLSH(threshold=0.5, num_perm=num_perm)

for user, movies in user_movies.items():
    m = MinHash(num_perm=num_perm)
    for movie in movies:
        m.update(str(movie).encode('utf8'))
    m_hashes[user] = m
    lsh.insert(user, m)


## top 100 similar pairs
similar_pairs = set()
for user, min_hash in m_hashes.items():
    result = lsh.query(min_hash)
    result.remove(user)  # Remove self-matching
    for other_user in result:
        if user != other_user:
            similar_pairs.add(tuple(sorted((user, other_user))))

top_100_pairs_with_jaccard = sorted(
    [(pair, m_hashes[pair[0]].jaccard(m_hashes[pair[1]])) for pair in similar_pairs],
    key=lambda x: x[1],
    reverse=True
)[:100]

top_100_pairs_with_jaccard

'''Partial Results:
[((72276, 111612), 1.0),
 ((35628, 169217), 1.0),
 ((14443, 18603), 1.0),
 ((234323, 264008), 1.0),
 ((27003, 299432), 1.0),
 ((113505, 326723), 1.0),
 ((76286, 99734), 1.0),
 ((68757, 92370), 1.0),
 ((92396, 208026), 1.0),
 ((138203, 312682), 1.0),
 ((9002, 255504), 1.0),
 ((109192, 289236), 1.0),
 ...]
'''




# Problem 2:
## Generate random 100 pairs
user_list = ratings_df['userId'].unique()

def generate_random_pairs(user_list, num_pairs):
    pairs = set()
    while len(pairs) < num_pairs:
        pair = tuple(np.random.choice(user_list, 2, replace=False))
        pairs.add(pair)
    return list(pairs)

random_pairs = generate_random_pairs(user_list, 100)
top_100_pairs = [pair[0] for pair in top_100_pairs_with_jaccard]

## Calculate average correlation
def calculate_average_correlation(pairs, ratings_df):
    correlations = []
    for user1, user2 in pairs:
        # Get the movies rated by both users
        user1_movies = ratings_df[ratings_df['userId'] == user1]
        user2_movies = ratings_df[ratings_df['userId'] == user2]
        common_movies = user1_movies[user1_movies['movieId'].isin(user2_movies['movieId'])]

        if not common_movies.empty:
            # Merge their ratings for the same movies
            merged_ratings = pd.merge(user1_movies, user2_movies, on="movieId")
            # Calculate Pearson correlation coefficient
            if len(merged_ratings) > 1:
                corr, _ = pearsonr(merged_ratings['rating_x'], merged_ratings['rating_y'])
                correlations.append(corr)

    # Filter out NaN values which can arise if users have rated the same movies but only one movie in common
    filtered_correlations = [corr for corr in correlations if not np.isnan(corr)]
    return np.mean(filtered_correlations) if filtered_correlations else 0


top_100_correlation = calculate_average_correlation(top_100_pairs, ratings_df)
average_correlation_random = calculate_average_correlation(random_pairs, ratings_df)

print("Average correlation of top 100 similar pairs:", top_100_correlation)
print("Average correlation of 100 random pairs:", average_correlation_random)

'''Results:
Average correlation of top 100 similar pairs: -0.023626062819918664
Average correlation of 100 random pairs: 0.11596791586208896
'''





# Problem 3: Train-Test-Validation Split
ratings = ratings_df.sample(frac=1, random_state=42)
train_set, temp_set = train_test_split(ratings, test_size=0.2, random_state=42)
validation_set, test_set = train_test_split(temp_set, test_size=0.5, random_state=42)

train_set.to_csv('train_set.csv', index=False)
validation_set.to_csv('validation_set.csv', index=False)
test_set.to_csv('test_set.csv', index=False)





# Problem 4: Popularity Baseline Model
ratings = pd.read_csv("full/ratings.csv")
train_set = pd.read_csv("full/train_set.csv")
test_set = pd.read_csv("full/test_set.csv")
validation_set = pd.read_csv("full/validation_set.csv")

train_popularity_df = train_set.groupby('movieId').size().reset_index(name='ratingCount')
train_popularity_df = train_popularity_df.merge(movies_df, on='movieId', how='left')
train_popularity_df = train_popularity_df.sort_values(by='ratingCount', ascending=False)

top_100_movies = train_popularity_df.head(100)['movieId'].tolist()

test_actuals = test_set.groupby('userId')['movieId'].agg(list)

user_predictions = {user: top_100_movies for user in test_actuals.keys()}


## Precision at k
def precision_at_k(actual, predicted, k):
    actual_set = set(actual)
    predicted_at_k = predicted[:k]
    hits = sum([1 for p in predicted_at_k if p in actual_set])
    return hits / k

## Mean Average Precision (MAP)
def mean_average_precision(actual, predicted):
    hit_list = [1 if p in actual else 0 for p in predicted]
    precision_at_k = [sum(hit_list[:k + 1]) / (k + 1) for k in range(len(hit_list))]
    relevant_count = sum(hit_list)
    if relevant_count > 0:
        return sum([x*y for x, y in zip(hit_list, precision_at_k)]) / relevant_count
    else:
        return 0

## NDCG at k
def ndcg_at_k(actual, predicted, k):
    def dcg(relevance_scores):
        return sum([rel / np.log2(idx + 2) for idx, rel in enumerate(relevance_scores)])

    actual_set = set(actual)
    predicted_at_k = predicted[:k]
    relevance_scores = [1 if p in actual_set else 0 for p in predicted_at_k]
    idcg = dcg(sorted(relevance_scores, reverse=True))
    dcg_score = dcg(relevance_scores)

    return dcg_score / idcg if idcg > 0 else 0

k = 100  
precisions = []
maps = []
ndcgs = []

for user, actual_movies in test_actuals.items():
    predicted_movies = user_predictions[user]

    precisions.append(precision_at_k(actual_movies, predicted_movies, k))
    maps.append(mean_average_precision(actual_movies, predicted_movies))
    ndcgs.append(ndcg_at_k(actual_movies, predicted_movies, k))

mean_precision = np.mean(precisions)
mean_map = np.mean(maps)
mean_ndcg = np.mean(ndcgs)

print(f"Mean Precision at {k}: {mean_precision}")
print(f"Mean Average Precision (MAP): {mean_map}")
print(f"Mean NDCG at {k}: {mean_ndcg}")


'''
Result on full dataset:
Mean Precision at 100: 0.020110841517150018
Mean Average Precision (MAP): 0.07823303725249714
Mean NDCG at 100: 0.22143836368919992
'''





# Problem 5: Recommendation System
spark = SparkSession.builder \
        .appName("ALSExample") \
        .config("spark.sql.shuffle.partitions", "200") \
        .getOrCreate()

file_path = 'hdfs:/user/jx1206_nyu_edu/ratings.csv'
ratings_df = spark.read.csv(file_path, header=True, inferSchema=True)

ratings_df = ratings_df.orderBy(rand())
train_df, temp_df = ratings_df.randomSplit([0.8, 0.2], seed=42)
validation_df, test_df = temp_df.randomSplit([0.5, 0.5], seed=42)

train_df.write.csv('hdfs:/user/jx1206_nyu_edu/train_data.csv', header=True, mode='overwrite')
validation_df.write.csv('hdfs:/user/jx1206_nyu_edu/val_data.csv', header=True, mode='overwrite')
test_df.write.csv('hdfs:/user/jx1206_nyu_edu/test_data.csv', header=True, mode='overwrite')

## Build the ALS model
als = ALS(maxIter=5, 
          userCol="userId", 
          itemCol="movieId", 
          ratingCol="rating", 
          coldStartStrategy="drop")
model = als.fit(train_df)


## Cross-Validation for Parameter Tuning
paramGrid = ParamGridBuilder() \
    .addGrid(als.rank, [10, 20, 50, 70]) \
    .addGrid(als.regParam, [0.01, 0.05, 0.1, 0.5]) \
    .build()

crossval = CrossValidator(estimator=als,
                          estimatorParamMaps=paramGrid,
                          evaluator=RegressionEvaluator(metricName="rmse", labelCol="rating"),
                          numFolds=5)  

cvModel = crossval.fit(train_df)

bestModel = cvModel.bestModel

print("Best model's rank:", bestModel._java_obj.parent().getRank())
print("Best model's maxIter:", bestModel._java_obj.parent().getMaxIter())
print("Best model's regParam:", bestModel._java_obj.parent().getRegParam())

'''
Best model's rank: 10
Best model's maxIter: 5
Best model's regParam: 0.1
'''

## RMSE
predictions = bestModel.transform(validation_df)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print("Root-mean-square error = " + str(rmse))

test_predictions = bestModel.transform(test_df)
test_rmse = evaluator.evaluate(test_predictions)
print("Test RMSE: ", test_rmse)

'''
Root-mean-square error = 0.8144966901198019
Test RMSE:  0.8136604712930299
'''


## Precision at k
def precision_at_k(actual, predicted, k):
    actual_set = set(actual)
    predicted_at_k = predicted[:k]
    hits = sum([1 for p in predicted_at_k if p in actual_set])
    return hits / k

## Mean Average Precision (MAP)
def mean_average_precision(actual, predicted):
    hit_list = [1 if p in actual else 0 for p in predicted]
    precision_at_k = [sum(hit_list[:k + 1]) / (k + 1) for k in range(len(hit_list))]
    relevant_count = sum(hit_list)
    if relevant_count > 0:
        return sum([x*y for x, y in zip(hit_list, precision_at_k)]) / relevant_count
    else:
        return 0

## NDCG at k
def ndcg_at_k(actual, predicted, k):
    def dcg(relevance_scores):
        return sum([rel / np.log2(idx + 2) for idx, rel in enumerate(relevance_scores)])
    
    actual_set = set(actual)
    predicted_at_k = predicted[:k]
    relevance_scores = [1 if p in actual_set else 0 for p in predicted_at_k]
    idcg = dcg(sorted(relevance_scores, reverse=True))
    dcg_score = dcg(relevance_scores)
    
    return dcg_score / idcg if idcg > 0 else 0


top_k = 100
user_recommendations = bestModel.recommendForAllUsers(top_k).collect()

user_predictions = {row['userId']: [row['recommendations'][i]['movieId'] for i in range(top_k)] for row in user_recommendations}

validation_actuals = validation_df.groupBy("userId").agg(collect_list("movieId").alias("actual")).rdd.collectAsMap()


precisions = []
maps = []
ndcgs = []

for user, actual_movies in validation_actuals.items():
    if user in user_predictions:
        predicted_movies = user_predictions[user]
        
        precisions.append(precision_at_k(actual_movies, predicted_movies, top_k))
        maps.append(mean_average_precision(actual_movies, predicted_movies))
        ndcgs.append(ndcg_at_k(actual_movies, predicted_movies, top_k))

mean_precision = np.mean(precisions)
mean_map = np.mean(maps)
mean_ndcg = np.mean(ndcgs)

print(f"Mean Precision at {top_k}: {mean_precision}")
print(f"Mean Average Precision (MAP): {mean_map}")
print(f"Mean NDCG at {top_k}: {mean_ndcg}")
spark.stop()

'''
Result on full dataset:
Mean Precision at 100:  1.0558020298878753e-05
Mean Average Precision (MAP):  2.157043423053422e-05
Mean NDCG at 100:  0.00017337283277620157
'''
