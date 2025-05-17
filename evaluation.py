import numpy as np
from sklearn.metrics import precision_score, recall_score, ndcg_score
import matplotlib.pyplot as plt
from tqdm import tqdm

def precision_at_k(model, test_df, n_users, n_items, user_features=None, item_features=None, k=5):
    """
    Calculate precision@k
    
    Args:
        model: Trained LightFM model
        test_df: Test DataFrame with user_idx, item_idx, label
        n_users: Number of users
        n_items: Number of items
        user_features: User features (optional)
        item_features: Item features (optional)
        k: Cutoff for precision calculation
        
    Returns:
        Precision@k score
    """
    precisions = []
    
    user_groups = test_df.groupby('user_idx')
    
    for user_idx, group in tqdm(user_groups, desc=f"Calculating precision@{k}", leave=False):
        pos_items = group[group['label'] == 1]['item_idx'].values
        
        if len(pos_items) == 0:
            continue
            
        all_items = group['item_idx'].values
        
        scores = model.predict(
            user_ids=np.repeat(user_idx, len(all_items)),
            item_ids=all_items,
            user_features=user_features,
            item_features=item_features
        )
        
        top_k_items = all_items[np.argsort(-scores)[:k]]
        
        precision = len(np.intersect1d(top_k_items, pos_items)) / k
        precisions.append(precision)
    
    return np.mean(precisions)

def recall_at_k(model, test_df, n_users, n_items, user_features=None, item_features=None, k=5):
    """
    Calculate recall@k
    
    Args:
        model: Trained LightFM model
        test_df: Test DataFrame with user_idx, item_idx, label
        n_users: Number of users
        n_items: Number of items
        user_features: User features (optional)
        item_features: Item features (optional)
        k: Cutoff for recall calculation
        
    Returns:
        Recall@k score
    """
    recalls = []
    
    user_groups = test_df.groupby('user_idx')
    
    for user_idx, group in tqdm(user_groups, desc=f"Calculating recall@{k}", leave=False):
        pos_items = group[group['label'] == 1]['item_idx'].values
        
        if len(pos_items) == 0:
            continue
            
        all_items = group['item_idx'].values
        
        scores = model.predict(
            user_ids=np.repeat(user_idx, len(all_items)),
            item_ids=all_items,
            user_features=user_features,
            item_features=item_features
        )
        
        top_k_items = all_items[np.argsort(-scores)[:k]]
        
        recall = len(np.intersect1d(top_k_items, pos_items)) / len(pos_items)
        recalls.append(recall)
    
    return np.mean(recalls)

def f1_at_k(model, test_df, n_users, n_items, user_features=None, item_features=None, k=5):
    """
    Calculate F1 score at k
    
    Args:
        model: Trained LightFM model
        test_df: Test DataFrame with user_idx, item_idx, label
        n_users: Number of users
        n_items: Number of items
        user_features: User features (optional)
        item_features: Item features (optional)
        k: Cutoff for F1 calculation
        
    Returns:
        F1@k score
    """
    f1_scores = []
    
    user_groups = test_df.groupby('user_idx')
    
    for user_idx, group in tqdm(user_groups, desc=f"Calculating F1@{k}", leave=False):
        pos_items = group[group['label'] == 1]['item_idx'].values
        
        if len(pos_items) == 0:
            continue
            
        all_items = group['item_idx'].values
        
        scores = model.predict(
            user_ids=np.repeat(user_idx, len(all_items)),
            item_ids=all_items,
            user_features=user_features,
            item_features=item_features
        )
        
        top_k_items = all_items[np.argsort(-scores)[:k]]
        
        n_relevant_and_recommended = len(np.intersect1d(top_k_items, pos_items))
        
        precision = n_relevant_and_recommended / k if k > 0 else 0
        recall = n_relevant_and_recommended / len(pos_items) if len(pos_items) > 0 else 0
        
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0
            
        f1_scores.append(f1)
    
    return np.mean(f1_scores)

def ndcg_at_k(model, test_df, n_users, n_items, user_features=None, item_features=None, k=10):
    """
    Calculate NDCG@k
    
    Args:
        model: Trained LightFM model
        test_df: Test DataFrame with user_idx, item_idx, label
        n_users: Number of users
        n_items: Number of items
        user_features: User features (optional)
        item_features: Item features (optional)
        k: Cutoff for NDCG calculation
        
    Returns:
        NDCG@k score
    """
    ndcgs = []
    
    user_groups = test_df.groupby('user_idx')
    
    for user_idx, group in tqdm(user_groups, desc=f"Calculating NDCG@{k}", leave=False):
        items = group['item_idx'].values
        labels = group['label'].values
        
        if sum(labels) == 0:
            continue
            
        scores = model.predict(
            user_ids=np.repeat(user_idx, len(items)),
            item_ids=items,
            user_features=user_features,
            item_features=item_features
        )
        
        sorted_indices = np.argsort(-scores)
        sorted_labels = labels[sorted_indices]
        
        sorted_labels = sorted_labels[:k]
        
        dcg = np.sum((2**sorted_labels - 1) / np.log2(np.arange(2, len(sorted_labels) + 2)))
        
        ideal_labels = np.sort(labels)[::-1][:k]
        idcg = np.sum((2**ideal_labels - 1) / np.log2(np.arange(2, len(ideal_labels) + 2)))
        
        if idcg > 0:
            ndcg = dcg / idcg
            ndcgs.append(ndcg)
    
    return np.mean(ndcgs)

def evaluate_model(model, test_df, n_users, n_items, user_features=None, item_features=None):
    """
    Evaluate a model on multiple metrics
    
    Args:
        model: Trained LightFM model
        test_df: Test DataFrame with user_idx, item_idx, label
        n_users: Number of users
        n_items: Number of items
        user_features: User features (optional)
        item_features: Item features (optional)
        
    Returns:
        Dictionary with evaluation metrics
    """
    metrics = {}
    
    # Calculate precision@k
    for k in [5, 10, 20, 50]:
        metrics[f'precision@{k}'] = precision_at_k(
            model, test_df, n_users, n_items, user_features, item_features, k
        )
        
        metrics[f'recall@{k}'] = recall_at_k(
            model, test_df, n_users, n_items, user_features, item_features, k
        )
        
        metrics[f'f1@{k}'] = f1_at_k(
            model, test_df, n_users, n_items, user_features, item_features, k
        )
        
        metrics[f'ndcg@{k}'] = ndcg_at_k(
            model, test_df, n_users, n_items, user_features, item_features, k
        )
    
    # Calculate coverage and diversity
    if hasattr(model, 'item_embeddings'):
        metrics['item_coverage@10'] = calculate_coverage(
            model, test_df, n_users, n_items, user_features, item_features, k=10
        )
        
        metrics['diversity@10'] = calculate_diversity(
            model, test_df, n_users, n_items, user_features, item_features, k=10
        )
    
    return metrics

def calculate_coverage(model, test_df, n_users, n_items, user_features=None, item_features=None, k=10):
    """
    Calculate the percentage of items that appear in recommendations for any user
    
    Args:
        model: Trained LightFM model
        test_df: Test DataFrame with user_idx, item_idx, label
        n_users: Number of users
        n_items: Number of items
        user_features: User features (optional)
        item_features: Item features (optional)
        k: Recommendation cutoff
        
    Returns:
        Coverage percentage
    """
    unique_users = test_df['user_idx'].unique()
    
    # Sample users for efficiency if there are too many
    if len(unique_users) > 200:
        unique_users = np.random.choice(unique_users, 200, replace=False)
        
    recommended_items = set()
    
    # Process in batches to handle large item counts
    batch_size = 1000
    
    for user_idx in tqdm(unique_users, desc=f"Calculating coverage@{k}", leave=False):
        # Process items in batches
        for start_idx in range(0, n_items, batch_size):
            end_idx = min(start_idx + batch_size, n_items)
            batch_items = np.arange(start_idx, end_idx)
            
            # Predict scores for this batch
            scores = model.predict(
                user_ids=np.repeat(user_idx, len(batch_items)),
                item_ids=batch_items,
                user_features=user_features,
                item_features=item_features
            )
            
            # Get top k items from this batch
            top_items_in_batch = batch_items[np.argsort(-scores)[:min(k, len(batch_items))]]
            
            # Add to set of all recommended items for this user
            for item in top_items_in_batch:
                recommended_items.add(item)
    
    # Calculate coverage
    coverage = len(recommended_items) / n_items
    return coverage

def calculate_diversity(model, test_df, n_users, n_items, user_features=None, item_features=None, k=10):
    """
    Calculate average pairwise distance between recommended items using item embeddings
    
    Args:
        model: Trained LightFM model
        test_df: Test DataFrame with user_idx, item_idx, label
        n_users: Number of users
        n_items: Number of items
        user_features: User features (optional)
        item_features: Item features (optional)
        k: Recommendation cutoff
        
    Returns:
        Average diversity score
    """
    if not hasattr(model, 'item_embeddings'):
        return 0.0
        
    unique_users = test_df['user_idx'].unique()
    diversity_scores = []
    
    # Get item embeddings
    item_embeddings = model.item_embeddings
    embedding_size = item_embeddings.shape[0]  # Correct dimension
    
    sampled_users = np.random.choice(unique_users, min(100, len(unique_users)), replace=False)
    
    for user_idx in tqdm(sampled_users, desc=f"Calculating diversity@{k}", leave=False):
        # Get all possible items but limit to embedding size
        all_items = np.arange(min(n_items, embedding_size))
        
        # Predict scores
        scores = model.predict(
            user_ids=np.repeat(user_idx, len(all_items)),
            item_ids=all_items,
            user_features=user_features,
            item_features=item_features
        )
        
        # Get top k items
        top_k_items = all_items[np.argsort(-scores)[:k]]
        
        if len(top_k_items) <= 1:
            continue
            
        # Calculate pairwise distances
        user_diversity = 0.0
        count = 0
        
        for i in range(len(top_k_items)):
            for j in range(i+1, len(top_k_items)):
                item_i = top_k_items[i]
                item_j = top_k_items[j]
                
                # Calculate cosine distance
                embedding_i = item_embeddings[item_i]
                embedding_j = item_embeddings[item_j]
                
                dot_product = np.dot(embedding_i, embedding_j)
                norm_i = np.linalg.norm(embedding_i)
                norm_j = np.linalg.norm(embedding_j)
                
                similarity = dot_product / (norm_i * norm_j) if norm_i > 0 and norm_j > 0 else 0
                distance = 1.0 - similarity
                
                user_diversity += distance
                count += 1
        
        if count > 0:
            user_diversity /= count
            diversity_scores.append(user_diversity)
    
    # Calculate average diversity
    if diversity_scores:
        return np.mean(diversity_scores)
    else:
        return 0.0

def plot_learning_curves(train_metrics, test_metrics, metric_names, epochs, model_name):
    """
    Plot learning curves for multiple metrics
    
    Args:
        train_metrics: List of training metrics
        test_metrics: List of test metrics
        metric_names: List of metric names
        epochs: List of epoch numbers
        model_name: Name of the model for plot title
    """
    n_metrics = len(metric_names)
    fig, axes = plt.subplots(1, n_metrics, figsize=(15, 5))
    
    for i, metric in enumerate(metric_names):
        ax = axes[i]
        ax.plot(epochs, [m[metric] for m in train_metrics], 'b-', label='Train')
        ax.plot(epochs, [m[metric] for m in test_metrics], 'r-', label='Test')
        ax.set_title(f'{metric} - {model_name}')
        ax.set_xlabel('Epochs')
        ax.set_ylabel(metric)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(f'{model_name}_learning_curves.png')
    plt.show() 
