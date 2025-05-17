import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.model_selection import train_test_split
from collections import defaultdict
from tqdm import tqdm
from sklearn.preprocessing import RobustScaler


def derive_implicit_labels(df, watch_ratio_threshold=0.7):
    """
    Ajoute une colonne 'label' (0/1) selon le seuil de watch_ratio.
    """
    df = df.copy()
    df['label'] = (df.get('watch_ratio', 0) >= watch_ratio_threshold).astype(int)
    return df


def filter_interactions(df, min_positive_interactions=3):
    """
    Conserve utilisateurs et vidéos ayant au moins N interactions positives.
    """
    pos = df[df['label'] == 1]
    user_counts = pos['user_id'].value_counts()
    item_counts = pos['video_id'].value_counts()

    users_ok = user_counts[user_counts >= min_positive_interactions].index
    items_ok = item_counts[item_counts >= min_positive_interactions].index

    df_filt = df[df['user_id'].isin(users_ok) & df['video_id'].isin(items_ok)]

    print(f"Interactions avant filtrage : {len(df)}, après : {len(df_filt)}")
    print(f"Utilisateurs gardés : {len(users_ok)}, vidéos gardées : {len(items_ok)}")

    return df_filt.reset_index(drop=True), list(users_ok), list(items_ok)


def create_user_item_maps(users, items):
    """
    Crée deux dictionnaires pour utilisateurs et vidéos.
    """
    user_to_idx = {uid: idx for idx, uid in enumerate(users)}
    idx_to_user = {idx: uid for uid, idx in user_to_idx.items()}

    item_to_idx = {vid: idx for idx, vid in enumerate(items)}
    idx_to_item = {idx: vid for vid, idx in item_to_idx.items()}

    return user_to_idx, idx_to_user, item_to_idx, idx_to_item


def leave_n_out_split(
    df, user_to_idx, item_to_idx,
    test_ratio=0.2, neg_ratio=4,
    test_neg_ratio=99, random_state=42
):
    """
    Split leave-n-out : pour chaque utilisateur, réserve un sous-ensemble pour test.
    """
    rng = np.random.RandomState(random_state)
    n_users = len(user_to_idx)
    n_items = len(item_to_idx)

    # Construire listes positives et inventaire complet
    pos_dict = defaultdict(list)
    all_dict = defaultdict(list)
    for _, row in df.iterrows():
        u = user_to_idx[row['user_id']]
        i = item_to_idx[row['video_id']]
        all_dict[u].append(i)
        if row['label'] == 1:
            pos_dict[u].append(i)

    train_records = []
    test_records = []

    for u in range(n_users):
        positives = pos_dict.get(u, [])
        if not positives:
            continue
        train_pos, test_pos = train_test_split(
            positives, test_size=test_ratio,
            random_state=random_state + u
        )
        # Ajouter positifs
        train_records += [(u, i, 1) for i in train_pos]
        test_records += [(u, i, 1) for i in test_pos]

        # Négatifs entraînement
        neg_candidates = list(set(range(n_items)) - set(all_dict[u]))
        n_train_neg = len(train_pos) * neg_ratio
        train_negs = (
            rng.choice(neg_candidates, n_train_neg, replace=False)
            if len(neg_candidates) >= n_train_neg
            else neg_candidates
        )
        train_records += [(u, i, 0) for i in train_negs]

        # Négatifs test
        n_test_neg = len(test_pos) * test_neg_ratio
        test_negs = (
            rng.choice(neg_candidates, n_test_neg, replace=False)
            if len(neg_candidates) >= n_test_neg
            else rng.choice(neg_candidates, n_test_neg, replace=True)
        )
        test_records += [(u, i, 0) for i in test_negs]

    # Construire DataFrames et matrices
    train_df = pd.DataFrame(train_records, columns=['user_idx', 'item_idx', 'label'])
    test_df = pd.DataFrame(test_records, columns=['user_idx', 'item_idx', 'label'])

    train_mat = sparse.coo_matrix(
        (train_df['label'] == 1).astype(int).values,
        (train_df['user_idx'], train_df['item_idx']),
        shape=(n_users, n_items)
    )
    test_mat = sparse.coo_matrix(
        (test_df['label'] == 1).astype(int).values,
        (test_df['user_idx'], test_df['item_idx']),
        shape=(n_users, n_items)
    )

    print(f"Train interactions: {len(train_df)}, Test interactions: {len(test_df)}")

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
    Crée une matrice CSR des catégories vidéo.
    """
    # Filtrer
    cats = item_categories_df[item_categories_df['video_id'].isin(item_to_idx)]

    # Ajouter inconnus
    missing = set(item_to_idx) - set(cats['video_id'])
    if missing:
        unknown = pd.DataFrame([{'video_id': mid, 'feat': 'unknown'} for mid in missing])
        cats = pd.concat([cats, unknown], ignore_index=True)

    # Découper catégories et indexer
    cats['feat_list'] = cats['feat'].str.split(',').apply(lambda L: [c.strip() or 'unknown' for c in L])
    all_feats = sorted({f for sub in cats['feat_list'] for f in sub})
    feat_idx = {f: i for i, f in enumerate(all_feats)}

    rows, cols, data = [], [], []
    for _, row in cats.iterrows():
        vidx = item_to_idx[row['video_id']]
        for f in row['feat_list']:
            fid = feat_idx[f]
            rows.append(vidx)
            cols.append(fid)
            data.append(1)

    mat = sparse.csr_matrix((data, (rows, cols)), shape=(len(item_to_idx), len(feat_idx)))
    # Normalisation L2
    norms = np.sqrt(mat.multiply(mat).sum(axis=1)).A1
    norms[norms == 0] = 1
    mat = sparse.diags(1 / norms) @ mat

    print(f"Item features matrix: {mat.shape}, sparsity: {1 - mat.nnz/(mat.shape[0]*mat.shape[1]):.4f}")
    return mat


def prepare_user_features(user_features_df, user_to_idx):
    """
    Crée une matrice CSR des caractéristiques utilisateur.
    """
    df = user_features_df.copy()
    df = df[df['user_id'].isin(user_to_idx)]

    # Colonnes numériques
    nums = [c for c in ['follow_user_num', 'fans_user_num', 'friend_user_num', 'register_days'] if c in df]
    df[nums] = df[nums].fillna(df[nums].median())
    if nums:
        df[nums] = RobustScaler().fit_transform(df[nums])

    # Catégorielles
    cats = [c for c in ['user_active_degree', 'follow_user_num_range',
                         'fans_user_num_range', 'friend_user_num_range', 'register_days_range'] if c in df]
    for c in cats:
        df[c] = df[c].fillna(df[c].mode().iloc[0])

    # Encodage manuel
    feat_map = {}
    idx = 0
    for c in nums:
        feat_map[c] = idx; idx += 1
    for c in cats:
        for val in df[c].unique():
            feat_map[f"{c}_{val}"] = idx; idx += 1

    rows, cols, data = [], [], []
    for _, row in df.iterrows():
        u = user_to_idx[row['user_id']]
        for c in nums:
            rows.append(u); cols.append(feat_map[c]); data.append(row[c])
        for c in cats:
            key = f"{c}_{row[c]}"
            rows.append(u); cols.append(feat_map[key]); data.append(1)

    mat = sparse.csr_matrix((data, (rows, cols)), shape=(len(user_to_idx), len(feat_map)))
    norms = np.sqrt(mat.multiply(mat).sum(axis=1)).A1
    norms[norms == 0] = 1
    mat = sparse.diags(1 / norms) @ mat

    print(f"User features matrix: {mat.shape}, sparsity: {1 - mat.nnz/(mat.shape[0]*mat.shape[1]):.4f}")
    return mat
