import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

def create_train_test_masks(ratings, split_ratio=0.8, seed=42):
    """
    Creates training and test masks by splitting nonzero ratings in the matrix.

    Parameters:
    - ratings: np.array (User-Movie Rating Matrix)
    - split_ratio: float (default 0.8) - fraction of nonzero entries to retain in train mask
    - seed: int (random seed for reproducibility)

    Returns:
    - train_matrix: np.array with 80% of ratings, 0 elsewhere
    - test_matrix: np.array with 20% of ratings, 0 elsewhere
    """
    np.random.seed(seed)

    # Find indices of all nonzero ratings
    nonzero_indices = np.argwhere(ratings > 0)  # Get (row, col) pairs

    # Shuffle indices
    np.random.shuffle(nonzero_indices)

    # Split indices into training (80%) and test (20%)
    split_idx = int(len(nonzero_indices) * split_ratio)
    train_indices = nonzero_indices[:split_idx]
    test_indices = nonzero_indices[split_idx:]

    # Create copies of the original matrix
    train_matrix = np.zeros_like(ratings)  # Empty training matrix
    test_matrix = np.zeros_like(ratings)   # Empty test matrix

    # Fill training matrix
    for row, col in train_indices:
        train_matrix[row, col] = ratings[row, col]

    # Fill test matrix
    for row, col in test_indices:
        test_matrix[row, col] = ratings[row, col]

    return train_matrix, test_matrix



def Jaccard_matrix(movies):
    """
    Compute the Jaccard distance matrix between movies based on their genres.

    Parameters:
    - movie_genres: pd.DataFrame (movie_id, genre1, genre2, ...)

    Returns:
    - similarity_matrix: np.array (n_movies, n_movies)
    """

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
            estimated_ratings[movie_id] = b_u[movie_id]  # Default to baseline if no neighbors exist

    return estimated_ratings


def fill_missing_ratings(user_reviews, distance_matrix, neigh_distance):
    """
    Returns a new user_ratings matrix where missing ratings (0s) are replaced
    with estimated ratings computed using get_movies_recommendations.
    """
    filled_ratings = user_reviews.copy()  # Copy as NumPy array
    pbar = tqdm(total=user_reviews.shape[0], desc="User Processed")

    for user_id in range(user_reviews.shape[0]):  # Iterate over users
        estimated_ratings = get_movies_recommendations(user_id, pd.DataFrame(user_reviews), distance_matrix, neigh_distance)

        for movie_id, rating in estimated_ratings.items():
            filled_ratings[user_id, movie_id] = rating  # Replace 0s with estimated ratings

        pbar.update(1)

    pbar.close()
    return pd.DataFrame(filled_ratings, columns=pd.DataFrame(user_reviews).columns, index=pd.DataFrame(user_reviews).index)


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
        predicted_ratings = fill_missing_ratings(R_train, distance_matrix, threshold)
        
        # Compute RMSE
        error = test_model(predicted_ratings, R_rest, R_val)
        print(f"RMSE: {error}")

        # Update the best model if this threshold gives a lower error
        if error < best_error:
            best_error = error
            best_threshold = threshold
            best_predictions = predicted_ratings.copy()

    print(f"\nBest threshold: {best_threshold} with RMSE: {best_error}")

    return best_threshold, best_error, best_predictions