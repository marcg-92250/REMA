import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.model_selection import train_test_split
from collections import defaultdict
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler

def derive_implicit_labels(df, watch_ratio_threshold=0.7):
    """
    Derive implicit feedback labels based on watch ratio
    
    Args:
        df: DataFrame with interaction data
        watch_ratio_threshold: Threshold for positive interactions
        
    Returns:
        DataFrame with added label column
    """
    df['label'] = (df['watch_ratio'] >= watch_ratio_threshold).astype(int)
    return df

def filter_interactions(df, min_positive_interactions=3):
    """
    Filter users and items with at least min_positive_interactions
    
    Args:
        df: DataFrame with interaction data and labels
        min_positive_interactions: Minimum positive interactions required
        
    Returns:
        Filtered DataFrame
    """
    print("Counting positive interactions per user and item...")
    user_pos_counts = df[df['label'] == 1]['user_id'].value_counts()
    item_pos_counts = df[df['label'] == 1]['video_id'].value_counts()
    
    valid_users = user_pos_counts[user_pos_counts >= min_positive_interactions].index
    valid_items = item_pos_counts[item_pos_counts >= min_positive_interactions].index
    
    print("Applying filtering...")
    filtered_df = df[df['user_id'].isin(valid_users) & df['video_id'].isin(valid_items)]
    
    print(f"Original interactions: {len(df)}")
    print(f"Filtered interactions: {len(filtered_df)}")
    print(f"Unique users: {len(valid_users)}")
    print(f"Unique items: {len(valid_items)}")
    
    return filtered_df, valid_users, valid_items

def create_user_item_maps(users, items):
    """
    Create mappings between IDs and indices
    
    Args:
        users: List of user IDs
        items: List of item IDs
        
    Returns:
        Tuple of mappings (user_to_idx, idx_to_user, item_to_idx, idx_to_item)
    """
    print("Creating user and item mappings...")
    user_to_idx = {u: i for i, u in enumerate(users)}
    idx_to_user = {i: u for i, u in enumerate(users)}
    item_to_idx = {i: j for j, i in enumerate(items)}
    idx_to_item = {j: i for j, i in enumerate(items)}
    
    return user_to_idx, idx_to_user, item_to_idx, idx_to_item

def leave_n_out_split(df, user_to_idx, item_to_idx, test_ratio=0.2, neg_ratio=4, test_neg_ratio=99, random_state=42):
    """
    Split data using leave-n-out strategy
    
    Args:
        df: DataFrame with interaction data
        user_to_idx: Mapping from user IDs to indices
        item_to_idx: Mapping from item IDs to indices
        test_ratio: Proportion of positive interactions to use for testing
        neg_ratio: Number of negative samples per positive in training
        test_neg_ratio: Number of negative samples per positive in testing
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary with train/test matrices and data
    """
    np.random.seed(random_state)
    
    print("Building user interaction dictionaries...")
    user_positives = defaultdict(list)
    user_all_items = defaultdict(list)
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing interactions"):
        user_idx = user_to_idx[row['user_id']]
        item_idx = item_to_idx[row['video_id']]
        
        user_all_items[user_idx].append(item_idx)
        if row['label'] == 1:
            user_positives[user_idx].append(item_idx)
    
    train_data = []
    test_data = []
    
    n_users = len(user_to_idx)
    n_items = len(item_to_idx)
    
    print("Creating train-test split...")
    for user_idx in tqdm(range(n_users), desc="Splitting users"):
        positives = user_positives[user_idx]
        
        if not positives:
            continue
        
        train_pos, test_pos = train_test_split(
            positives, test_size=test_ratio, random_state=random_state + user_idx
        )
        
        for item_idx in train_pos:
            train_data.append((user_idx, item_idx, 1))
        
        all_items = set(range(n_items))
        known_items = set(user_all_items[user_idx])
        unknown_items = list(all_items - known_items)
        
        if unknown_items:
            n_train_neg = len(train_pos) * neg_ratio
            if len(unknown_items) > n_train_neg:
                train_neg = np.random.choice(unknown_items, size=n_train_neg, replace=False)
            else:
                train_neg = unknown_items
                
            for item_idx in train_neg:
                train_data.append((user_idx, item_idx, 0))
        
        for item_idx in test_pos:
            test_data.append((user_idx, item_idx, 1))
            
        if unknown_items:
            n_test_neg = len(test_pos) * test_neg_ratio
            
            if len(unknown_items) > n_test_neg:
                test_neg = np.random.choice(unknown_items, size=n_test_neg, replace=False)
            else:
                test_neg = np.random.choice(unknown_items, size=n_test_neg, replace=True)
                
            for item_idx in test_neg:
                test_data.append((user_idx, item_idx, 0))
    
    print("Creating DataFrames and matrices...")
    train_df = pd.DataFrame(train_data, columns=['user_idx', 'item_idx', 'label'])
    test_df = pd.DataFrame(test_data, columns=['user_idx', 'item_idx', 'label'])
    
    print("Building sparse matrices...")
    train_mat = sparse.coo_matrix(
        (np.ones(len(train_df[train_df['label'] == 1])), 
         (train_df[train_df['label'] == 1]['user_idx'], train_df[train_df['label'] == 1]['item_idx'])),
        shape=(n_users, n_items)
    )
    
    test_mat = sparse.coo_matrix(
        (np.ones(len(test_df[test_df['label'] == 1])), 
         (test_df[test_df['label'] == 1]['user_idx'], test_df[test_df['label'] == 1]['item_idx'])),
        shape=(n_users, n_items)
    )
    
    print(f"Training interactions: {len(train_df)}")
    print(f"Testing interactions: {len(test_df)}")
    
    return {
        'train_interactions': train_mat,
        'test_interactions': test_mat,
        'train_df': train_df,
        'test_df': test_df,
        'n_users': n_users,
        'n_items': n_items
    }

def prepare_item_features(item_categories_df, item_to_idx):
    """
    Prepare item features for LightFM
    
    Args:
        item_categories_df: DataFrame with item categories
        item_to_idx: Mapping from item IDs to indices
        
    Returns:
        Sparse matrix with item features
    """
    from sklearn.preprocessing import OneHotEncoder
    from scipy import sparse
    import numpy as np
    
    print("Preparing item features...")
    item_cats = item_categories_df.copy()
    
    item_cats = item_cats[item_cats['video_id'].isin(item_to_idx.keys())]
    
    # Add a default category for items without categories
    missing_items = set(item_to_idx.keys()) - set(item_cats['video_id'])
    if missing_items:
        print(f"Adding default category for {len(missing_items)} items without categories")
        default_rows = []
        for item_id in missing_items:
            default_rows.append({'video_id': item_id, 'feat': 'unknown'})
        
        if default_rows:
            default_df = pd.DataFrame(default_rows)
            item_cats = pd.concat([item_cats, default_df])
    
    item_features = []
    item_indices = []
    feature_indices = []
    
    print("Extracting categories...")
    all_categories = set()
    for feat in tqdm(item_cats['feat'], desc="Processing categories"):
        if isinstance(feat, str):
            categories = [cat.strip() for cat in feat.split(',')]
            all_categories.update(categories)
    
    # Add unknown category
    all_categories.add('unknown')
    
    category_to_idx = {cat: i for i, cat in enumerate(sorted(all_categories))}
    
    print(f"Total unique categories: {len(category_to_idx)}")
    
    for idx, row in tqdm(item_cats.iterrows(), total=len(item_cats), desc="Building feature matrix"):
        if row['video_id'] not in item_to_idx:
            continue
            
        item_idx = item_to_idx[row['video_id']]
        
        # Extract categories from the feat column
        if isinstance(row['feat'], str):
            categories = [cat.strip() for cat in row['feat'].split(',')]
            
            # If no valid categories, add unknown
            if not categories or not any(cat in category_to_idx for cat in categories):
                item_indices.append(item_idx)
                feature_indices.append(category_to_idx['unknown'])
                item_features.append(1.0)
            else:
                for cat in categories:
                    if cat in category_to_idx:
                        item_indices.append(item_idx)
                        feature_indices.append(category_to_idx[cat])
                        item_features.append(1.0)
        else:
            # For items with no feat value, add unknown
            item_indices.append(item_idx)
            feature_indices.append(category_to_idx['unknown'])
            item_features.append(1.0)
    
    print("Creating sparse feature matrix...")
    n_items = len(item_to_idx)
    n_features = len(category_to_idx)
    
    # Create sparse matrix
    item_features_mat = sparse.csr_matrix(
        (item_features, (item_indices, feature_indices)),
        shape=(n_items, n_features)
    )
    
    # Use softer normalization - L2 norm instead of sum-to-1
    # This preserves more information while still preventing dominance
    for i in range(n_items):
        row = item_features_mat.getrow(i)
        norm = np.sqrt((row.multiply(row)).sum())
        if norm > 0:
            item_features_mat[i] = row / norm
    
    print(f"Item feature matrix shape: {item_features_mat.shape}")
    sparsity = 1.0 - item_features_mat.nnz / (item_features_mat.shape[0] * item_features_mat.shape[1])
    print(f"Matrix sparsity: {sparsity:.4f}")
    
    return item_features_mat

def prepare_user_features(user_features_df, user_to_idx):
    """
    Prepare user features for LightFM
    
    Args:
        user_features_df: DataFrame with user features
        user_to_idx: Mapping from user IDs to indices
        
    Returns:
        Sparse matrix with user features
    """
    print("Preparing user features...")
    user_feats = user_features_df.copy()
    
    # Keep only users in the user_to_idx mapping
    user_feats = user_feats[user_feats['user_id'].isin(user_to_idx.keys())].copy()
    
    # Add missing users with default values
    missing_users = set(user_to_idx.keys()) - set(user_feats['user_id'])
    if missing_users:
        print(f"Adding default features for {len(missing_users)} missing users")
        default_rows = []
        for user_id in missing_users:
            default_row = {'user_id': user_id}
            default_rows.append(default_row)
        
        if default_rows:
            default_df = pd.DataFrame(default_rows)
            user_feats = pd.concat([user_feats, default_df])
    
    # Extract numerical features that actually exist in the DataFrame
    numerical_features = []
    for col in ['follow_user_num', 'fans_user_num', 'friend_user_num', 'register_days']:
        if col in user_feats.columns:
            numerical_features.append(col)
    
    print(f"Using numerical features: {numerical_features}")
    
    # Fill NaNs with median values (more robust than mean)
    for col in numerical_features:
        if col in user_feats.columns:
            median = user_feats[col].median()
            user_feats[col] = user_feats[col].fillna(median)
    
    # Use RobustScaler to handle outliers better
    scaler = RobustScaler()
    if len(numerical_features) > 0:
        # Make sure all columns in numerical_features exist in the DataFrame
        existing_numerical = [col for col in numerical_features if col in user_feats.columns]
        if existing_numerical:
            user_feats[existing_numerical] = scaler.fit_transform(user_feats[existing_numerical])
    
    # Add categorical features
    categorical_features = []
    for col in ['user_active_degree', 'follow_user_num_range', 'fans_user_num_range',
                'friend_user_num_range', 'register_days_range']:
        if col in user_feats.columns:
            categorical_features.append(col)
    
    print(f"Using categorical features: {categorical_features}")
    
    # Fill categorical NaNs with most common value
    for col in categorical_features:
        if col in user_feats.columns:
            most_common = user_feats[col].mode()[0]
            user_feats[col] = user_feats[col].fillna(most_common)
            
    # Create sparse feature matrix
    user_indices = []
    feature_indices = []
    feature_values = []
    
    feature_to_idx = {}
    feature_idx = 0
    
    # Add numerical features
    for col in numerical_features:
        if col in user_feats.columns:
            feature_to_idx[col] = feature_idx
            feature_idx += 1
    
    # Add categorical features mappings
    for col in categorical_features:
        if col in user_feats.columns:
            for val in user_feats[col].dropna().unique():
                feature_name = f"{col}_{val}"
                feature_to_idx[feature_name] = feature_idx
                feature_idx += 1
    
    # Build feature matrix
    for _, row in user_feats.iterrows():
        if row['user_id'] not in user_to_idx:
            continue
            
        user_idx = user_to_idx[row['user_id']]
        
        # Add numerical features
        for col in numerical_features:
            if col in user_feats.columns and not pd.isna(row[col]):
                user_indices.append(user_idx)
                feature_indices.append(feature_to_idx[col])
                feature_values.append(float(row[col]))
        
        # Add categorical features
        for col in categorical_features:
            if col in user_feats.columns and not pd.isna(row[col]):
                feature_name = f"{col}_{row[col]}"
                if feature_name in feature_to_idx:
                    user_indices.append(user_idx)
                    feature_indices.append(feature_to_idx[feature_name])
                    feature_values.append(1.0)
    
    # Create sparse matrix
    n_users = len(user_to_idx)
    n_features = len(feature_to_idx)
    
    user_features_mat = sparse.csr_matrix(
        (feature_values, (user_indices, feature_indices)),
        shape=(n_users, n_features)
    )
    
    # L2 normalization for consistent scaling
    for i in range(n_users):
        row = user_features_mat.getrow(i)
        norm = np.sqrt((row.multiply(row)).sum())
        if norm > 0:
            user_features_mat[i] = row / norm
    
    print(f"User feature matrix shape: {user_features_mat.shape}")
    sparsity = 1.0 - user_features_mat.nnz / (user_features_mat.shape[0] * user_features_mat.shape[1])
    print(f"Matrix sparsity: {sparsity:.4f}")
    
    return user_features_mat 
