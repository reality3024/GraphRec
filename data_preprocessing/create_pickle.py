import scipy.io
import numpy as np
import pickle
from collections import defaultdict
import random
import os
from tqdm import tqdm
import pandas as pd

def create_user_product_map(arr, key_column, value_column):
    """
    Create a mapping of user IDs to the products they interact with.
    Returns a defaultdict of lists.
    """
    user_product_map = defaultdict(list)
    for row in arr:
        user_id = int(row[key_column])  # Convert to Python int
        product_id = int(row[value_column])  # Convert to Python int
        user_product_map[user_id].append(product_id)
    
    # Return as defaultdict without converting to dict
    return user_product_map

def create_product_user_map(arr, key_column, value_column):
    """
    Create a mapping of products to the user IDs they interact with and their ratings.
    Returns two defaultdicts: one for users and one for ratings.
    """
    product_user_map = defaultdict(list)
    product_rating_map = defaultdict(list)
    
    # Create a lookup map for (productID, userID) -> rating
    rating_lookup = {}
    for row in tqdm(arr, desc="Creating rating lookup"):
        product_id = int(row[key_column])  # Convert to Python int
        user_id = int(row[value_column])  # Convert to Python int
        rating = float(row[2])  # Convert to Python float
        rating_lookup[(product_id, user_id)] = rating
    
    # First pass: collect all user IDs for each product
    for row in arr:
        product_id = int(row[key_column])  # Convert to Python int
        user_id = int(row[value_column])  # Convert to Python int
        product_user_map[product_id].append(user_id)
    
    # Convert to defaultdict of sets to remove duplicates
    product_user_set_map = defaultdict(list)
    for product_id, users in product_user_map.items():
        product_user_set_map[product_id] = users
    
    # Second pass: collect ratings using the lookup map
    for product_id, user_set in tqdm(product_user_set_map.items(), desc="Processing products"):
        # Get ratings in the same order as users
        ratings = [rating_lookup[(product_id, user_id)] for user_id in user_set]
        product_rating_map[product_id] = ratings
    
    return product_user_set_map, product_rating_map

def check_missing_sequence(unique_users):
    # Check for missing numbers in unique_users sequence
    min_user = min(unique_users)
    max_user = max(unique_users)
    all_numbers = set(range(min_user, max_user + 1))
    missing_numbers = sorted(list(all_numbers - set(unique_users)))
    
    if missing_numbers:
        print(f"Missing user IDs in sequence: {missing_numbers}")
    else:
        print("No missing user IDs in sequence")
        
def create_social_adj_lists(trust_data):
    """
    Create social adjacency lists from trust data.
    Each key represents a user ID, and the value is a set of user IDs they trust.
    """
    social_adj_lists = defaultdict(set)
    
    for row in trust_data:
        user_id = int(row[0])
        trusted_user = int(row[1])
        social_adj_lists[user_id].add(trusted_user)
    
    # Convert to regular dict and sort by key
    return dict(sorted(social_adj_lists.items()))

def create_rating_list(arr):
    rating_list_map = dict()
    all_rating = arr[:,2]
    unique_rating = set(all_rating)
    rating_list_map[0.0] = 0
    for rating in unique_rating:
        rating = int(rating)
        rating_list_map[float(rating)] = rating
    return rating_list_map

def create_training_testing_data(all_ids, ratings_list, train_ratio=0.8, val_ratio=0.1):
    """
    Create training, validation and testing data from all_ids using ratings_list for rating value mapping.
    Returns train_u, train_v, train_r, val_u, val_v, val_r, test_u, test_v, test_r lists.
    """
    # Convert all_ids to appropriate format with explicit type conversion
    user_ids = np.array(all_ids[:, 0], dtype=np.int64)
    item_ids = np.array(all_ids[:, 1], dtype=np.int64)
    rating_indices = np.array(all_ids[:, 2], dtype=np.int64)
    
    # Create list of (user, item, rating) tuples
    all_ratings = []
    for i in range(len(all_ids)):
        user_id = int(user_ids[i])  # Convert to Python int
        item_id = int(item_ids[i])  # Convert to Python int
        rating_index = int(rating_indices[i])  # Convert to Python int
        # Look up actual rating value from ratings_list and convert to float
        rating_value = float(ratings_list[rating_index])
        all_ratings.append((user_id, item_id, rating_value))
    
    # Shuffle and split into train/val/test
    random.shuffle(all_ratings)
    train_idx = int(len(all_ratings) * train_ratio)
    val_idx = int(len(all_ratings) * (train_ratio + val_ratio))
    
    train_data = all_ratings[:train_idx]
    val_data = all_ratings[train_idx:val_idx]
    test_data = all_ratings[val_idx:]
    
    # Separate into user, item, rating lists
    train_u, train_v, train_r = zip(*train_data)
    val_u, val_v, val_r = zip(*val_data)
    test_u, test_v, test_r = zip(*test_data)
    
    # Convert to lists
    train_u = list(train_u)
    train_v = list(train_v)
    train_r = list(train_r)
    val_u = list(val_u)
    val_v = list(val_v)
    val_r = list(val_r)
    test_u = list(test_u)
    test_v = list(test_v)
    test_r = list(test_r)
    
    return train_u, train_v, train_r, val_u, val_v, val_r, test_u, test_v, test_r

def process_dataset(rating_file, trust_file, output_dir, train_ratio=0.8):
    # Load MATLAB files and print available keys
    print("Loading rating file...")
    rating_data = scipy.io.loadmat(rating_file)
    print(f"Available keys in rating file: {list(rating_data.keys())}")
    
    print("Loading trust file...")
    trust_data = scipy.io.loadmat(trust_file)
    print(f"Available keys in trust file: {list(trust_data.keys())}")
    
    # Get the actual data arrays
    # The actual key might be different, so we'll try to find the right one
    ratings = None
    for key in rating_data.keys():
        if not key.startswith('__'):  # Skip MATLAB internal variables
            if isinstance(rating_data[key], np.ndarray) and rating_data[key].shape[1] >= 4:
                ratings = rating_data[key]
                print(f"Using key '{key}' for ratings")
                break
    
    trust = None
    for key in trust_data.keys():
        if not key.startswith('__'):  # Skip MATLAB internal variables
            if isinstance(trust_data[key], np.ndarray) and trust_data[key].shape[1] == 2:
                trust = trust_data[key]
                print(f"Using key '{key}' for trust network")
                break
    
    if ratings is None:
        raise ValueError("Could not find valid ratings data in the MATLAB file")
    if trust is None:
        raise ValueError("Could not find valid trust network data in the MATLAB file")
    
    # Initialize data structures
    history_u_lists = defaultdict(list)
    history_ur_lists = defaultdict(list)
    history_v_lists = defaultdict(list)
    history_vr_lists = defaultdict(list)
    history_timestamp_lists = defaultdict(list)

    # Create all information from source data
    all_ids = ratings.astype(int)  # Convert ratings to appropriate format
    all_columns = list(range(all_ids.shape[1]))
    # Remove the specified columns(Category, helpfulness, timestamp)
    keep_columns = [col for col in all_columns if col not in [2,4]]
    # Return array with only the kept columns
    all_ids = all_ids[:, keep_columns]

    # Create mappings for users and items
    unique_users = np.unique(all_ids[:, 0])
    unique_items = np.unique(all_ids[:, 1])
    check_missing_sequence(unique_users)
    check_missing_sequence(unique_items)
    
    # Add missing data
    missing_data = np.array([
        [0, 0, 3, 1999999999],
        [20819, 0, 3, 1999999999],
        [21354, 0, 3, 1999999999]
    ])
    all_ids = np.vstack([all_ids, missing_data])

    # Export all_ids to CSV
    df = pd.DataFrame(all_ids, columns=['user_id', 'product_id', 'rating', 'timestamp'])
    csv_output_path = os.path.join(output_dir, 'mapped_data.csv')
    df.to_csv(csv_output_path, index=False)
    print(f"Exported data to {csv_output_path}")

    # Create rating list
    ratings_list = create_rating_list(all_ids)

    # Create training and testing data
    train_u, train_v, train_r, val_u, val_v, val_r, test_u, test_v, test_r = create_training_testing_data(all_ids, ratings_list, train_ratio)
    
    # Create history lists
    history_u_lists = create_user_product_map(all_ids, 0, 1)
    history_ur_lists = create_user_product_map(all_ids, 0, 2)
    history_v_lists, history_vr_lists = create_product_user_map(all_ids, 1, 0)
    history_vr_lists = create_user_product_map(all_ids, 1, 2)
    history_timestamp_lists = create_user_product_map(all_ids, 0, 3)
    
    # Process trust network (no need for mapping since we're using original IDs)
    social_adj_lists = create_social_adj_lists(trust)
    
    # Save processed data
    processed_data = (
        history_u_lists,
        history_ur_lists,
        history_v_lists,
        history_vr_lists,
        train_u,
        train_v,
        train_r,
        val_u,
        val_v,
        val_r,
        test_u,
        test_v,
        test_r,
        social_adj_lists,
        ratings_list,
        history_timestamp_lists
    )
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    dataset_name = 'Epinions'
    output_file = os.path.join(output_dir, f"dataset_{dataset_name}_train80val10test0.pickle")
    
    with open(output_file, 'wb') as f:
        pickle.dump(processed_data, f)


if __name__ == "__main__":
    rating_file = r'D:\workspace\GraphRec\data\Epinions\rating_with_timestamp.mat'
    trust_file = r'D:\workspace\GraphRec\data\Epinions\trust.mat'
    output_dir = r'D:\workspace\GraphRec\data\Epinions'
    process_dataset(rating_file, trust_file, output_dir) 
