import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import numpy as np
from numpy.linalg import solve
import tensorflow as tf
from tensorflow.keras.layers import Embedding, Flatten, Dense, Concatenate, Input
from tensorflow.keras.models import Model

def create_train_test_masks(ratings, split_ratio=0.8, seed=42):

    np.random.seed(seed)

    nonzero_indices = np.argwhere(ratings > 0)  
    np.random.shuffle(nonzero_indices)

    split_idx = int(len(nonzero_indices) * split_ratio)
    train_indices = nonzero_indices[:split_idx]
    test_indices = nonzero_indices[split_idx:]

    train_matrix = np.zeros_like(ratings)  
    test_matrix = np.zeros_like(ratings)   

    for row, col in train_indices:
        train_matrix[row, col] = ratings[row, col]

    for row, col in test_indices:
        test_matrix[row, col] = ratings[row, col]

    return train_matrix, test_matrix


def Jaccard_matrix(movies):

    A = movies.values[:, np.newaxis]  
    B = movies.values

    intersection = np.logical_and(A, B).sum(axis=2)
    union = np.logical_or(A, B).sum(axis=2)
    similarity_matrix = (intersection / np.maximum(union, 1))

    mean_distance = np.mean(similarity_matrix)
    max_distance = np.max(similarity_matrix)
    min_distance = np.min(similarity_matrix)
    #print(mean_distance, max_distance, min_distance)

    return similarity_matrix


def get_nearest_movies(movie_id, similarity_matrix, neigh_distance):

    mask = similarity_matrix[movie_id] >= neigh_distance
    output = np.where(np.array(mask) == 1)[0].tolist()
    return output


def get_movies_recommendations(user, user_reviews, similarity_matrix, neigh_distance):

    candidate_movies = user_reviews.columns[user_reviews.iloc[user] == 0].to_numpy()
    
    b_u = np.nanmean(user_reviews.replace(0, np.nan), axis=0)
    estimated_ratings = {}
    user_ratings = user_reviews.to_numpy()

    for movie_id in candidate_movies:
        output = get_nearest_movies(movie_id, similarity_matrix, neigh_distance)
        
        N_u = np.array([movie for movie in output if user_ratings[user, movie] > 0])

        if N_u.size > 0:
            numerator = np.dot(similarity_matrix[movie_id, N_u], (user_ratings[user, N_u] - b_u[N_u]))
            denominator = np.sum(similarity_matrix[movie_id, N_u])

            estimated_ratings[movie_id] = b_u[movie_id] + (numerator / denominator if denominator != 0 else 0)
        else:
            estimated_ratings[movie_id] = b_u[movie_id]  

    return estimated_ratings


def knn_estimation(user_reviews, distance_matrix, neigh_distance):

    filled_ratings = user_reviews.copy() 
    pbar = tqdm(total=user_reviews.shape[0], desc='Users Processed:')
    for user_id in range(user_reviews.shape[0]):  
        estimated_ratings = get_movies_recommendations(user_id, pd.DataFrame(user_reviews), distance_matrix, neigh_distance)
        for movie_id, rating in estimated_ratings.items():
            filled_ratings[user_id, movie_id] = rating  
        pbar.update(1)
    pbar.close()
    return pd.DataFrame(filled_ratings, columns=pd.DataFrame(user_reviews).columns, index=pd.DataFrame(user_reviews).index)

def train_latent_model(latent_dim, reg_param, max_iters, R_train, R_val):
    
    n_users, n_movies = R_train.shape    
    
    # We initialize U and M randomly   
    U= np.random.rand(n_users, latent_dim)      
    M = np.random.rand(n_movies, latent_dim)

    best_val_score = 100
    counter = 0
    
    for iteration in range(max_iters):
        # Solving for U while keeping M fixed
        for u in range(n_users):
            relevant_movies = R_train[u, :].nonzero()[0]    # Set of movies rated by user u
            if len(relevant_movies) > 0:
                M_subset = M[relevant_movies]               # Extract only relevant movie embeddings
                R_u = R_train[u, relevant_movies]           

                # Solving the least squares problem
                U[u] = solve(M_subset.T @ M_subset + reg_param * np.eye(latent_dim), M_subset.T @ R_u)

        # Solving for M while keeping U fixed
        for m in range(n_movies):
            relevant_users = R_train[:, m].nonzero()[0]     # Set of users who rated movie m
            if len(relevant_users) > 0:
                U_subset = U[relevant_users]                # Extract relevant user embeddings
                R_m = R_train[relevant_users, m] 

                # Solving least squares problem
                M[m] = solve(U_subset.T @ U_subset + reg_param * np.eye(latent_dim), U_subset.T @ R_m)

        if iteration % 50 == 0:
            
            # Computing RMSE
            predicted_ratings = U @ M.T
            train_mask = R_train > 0        
            val_mask = R_val > 0
            train_error = np.sqrt(np.sum((R_train - predicted_ratings) ** 2 * train_mask) / np.sum(train_mask))
            val_error = np.sqrt(np.sum((R_val - predicted_ratings) ** 2 * val_mask) / np.sum(val_mask))
            print(f"Iteration {iteration}, Train RMSE: {train_error:.4f}, Val RMSE: {val_error:.4f}")
            if val_error < best_val_score:
                # Updating best val score
                best_val_score = val_error
                counter = 0
            else:
                counter += 1
        
        if counter == 5:
            # 25 epochs without val improvements. We stop
            print('early stopping')
            break
        
    print('Training Finished')
    return U, M


def build_latent_nn(n_users, n_movies, latent_dim=10):
    
    # Input Layers
    user_input = Input(shape=(1,), name="user_input")
    movie_input = Input(shape=(1,), name="movie_input")

    # Embedding Layers 
    user_embedding = Embedding(n_users, latent_dim, name="user_embedding")(user_input)
    movie_embedding = Embedding(n_movies, latent_dim, name="movie_embedding")(movie_input)

    user_vector = Flatten()(user_embedding)
    movie_vector = Flatten()(movie_embedding)

    # GMF (Generalized Matrix Factorization) Component
    gmf_output = tf.keras.layers.multiply([user_vector, movie_vector])  

    # Fully Connected Layers
    fc_input = Concatenate()([user_vector, movie_vector])
    fc_hidden = Dense(64, activation='relu')(fc_input)
    fc_hidden = Dense(32, activation='relu')(fc_hidden)
    fc_hidden = Dense(16, activation='relu')(fc_hidden)

    fc_output = Concatenate()([gmf_output, fc_hidden])

    # Final prediction layer. We predict a single rating in [0, 1]
    output = Dense(1, activation='sigmoid', name="output_layer")(fc_output)

    # Build model
    model = Model(inputs=[user_input, movie_input], outputs=output)
    model.compile(optimizer='adam', loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError()])

    return model

def test_model(predicted_ratings, R, R_test):

    test_mask = R_test > 0
    test_error = np.sqrt(((R - predicted_ratings) ** 2 * test_mask).sum().sum() / np.sum(test_mask))

    return test_error


def optimal_knn_model(threshold_values, R_rest , R_train, R_val, G):

    best_threshold = None
    best_error = float("inf") 
    best_predictions = None

    for threshold in threshold_values:
        print(f"Testing threshold: {threshold}")

        distance_matrix = Jaccard_matrix(G)  
        predicted_ratings = knn_estimation(R_train, distance_matrix, threshold)
        
        error = test_model(predicted_ratings, R_rest, R_val)
        print(f"RMSE: {error}")

        if error < best_error:
            best_error = error
            best_threshold = threshold
            best_predictions = predicted_ratings.copy()

    print(f"\nBest threshold: {best_threshold} with RMSE: {best_error}")

    return best_threshold, best_error, best_predictions

