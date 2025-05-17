import numpy as np
from sklearn.metrics import precision_score, recall_score, ndcg_score
import matplotlib.pyplot as plt
from tqdm import tqdm


def precision_at_k(model, test_df, n_users, n_items, user_features=None, item_features=None, k=5):
    """
    Précision moyenne @k pour chaque utilisateur.
    Retourne un float.
    """
    def user_precision(df):
        pos = df[df['label'] == 1]['item_idx'].values
        if len(pos) == 0:
            return None
        candidates = df['item_idx'].values
        scores = model.predict(
            user_ids=np.full(len(candidates), df.name),
            item_ids=candidates,
            user_features=user_features,
            item_features=item_features
        )
        topk = candidates[np.argsort(-scores)[:k]]
        return np.intersect1d(topk, pos).size / k

    grp = test_df.groupby('user_idx')
    vals = [p for _, p in grp.apply(user_precision).dropna().items()]
    return np.mean(vals) if vals else 0.0


def recall_at_k(model, test_df, n_users, n_items, user_features=None, item_features=None, k=5):
    """
    Rappel moyen @k pour chaque utilisateur.
    Retourne un float.
    """
    def user_recall(df):
        pos = df[df['label'] == 1]['item_idx'].values
        if len(pos) == 0:
            return None
        candidates = df['item_idx'].values
        scores = model.predict(
            user_ids=np.full(len(candidates), df.name),
            item_ids=candidates,
            user_features=user_features,
            item_features=item_features
        )
        topk = candidates[np.argsort(-scores)[:k]]
        return np.intersect1d(topk, pos).size / len(pos)

    grp = test_df.groupby('user_idx')
    vals = [r for _, r in grp.apply(user_recall).dropna().items()]
    return np.mean(vals) if vals else 0.0


def f1_at_k(model, test_df, n_users, n_items, user_features=None, item_features=None, k=5):
    """
    F1 moyen @k pour chaque utilisateur.
    Retourne un float.
    """
    def user_f1(df):
        p = precision_at_k(model, df.reset_index(), n_users, n_items, user_features, item_features, k)
        r = recall_at_k(model, df.reset_index(), n_users, n_items, user_features, item_features, k)
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    grp = test_df.groupby('user_idx')
    vals = [f for _, f in grp.apply(user_f1).items()]
    return np.mean(vals) if vals else 0.0


def ndcg_at_k(model, test_df, n_users, n_items, user_features=None, item_features=None, k=10):
    """
    NDCG moyen @k pour chaque utilisateur.
    Retourne un float.
    """
    def user_ndcg(df):
        y_true = df.set_index('item_idx')['label']
        y_scores = pd.Series(
            model.predict(
                user_ids=np.full(len(y_true), df.name),
                item_ids=y_true.index.values,
                user_features=user_features,
                item_features=item_features
            ),
            index=y_true.index
        )
        return ndcg_score([y_true.values], [y_scores.values], k=k)

    grp = test_df.groupby('user_idx')
    vals = [n for _, n in grp.apply(user_ndcg).items()]
    return np.mean(vals) if vals else 0.0


def evaluate_model(model, test_df, n_users, n_items, user_features=None, item_features=None):
    """
    Retourne un dict de metrics pour k in [5,10,20,50] incluant precision, recall, f1, ndcg.
    """
    ks = [5, 10, 20, 50]
    res = {}
    for k in ks:
        res.update({
            f'precision@{k}': precision_at_k(model, test_df, n_users, n_items, user_features, item_features, k),
            f'recall@{k}': recall_at_k(model, test_df, n_users, n_items, user_features, item_features, k),
            f'f1@{k}': f1_at_k(model, test_df, n_users, n_items, user_features, item_features, k),
            f'ndcg@{k}': ndcg_at_k(model, test_df, n_users, n_items, user_features, item_features, k)
        })
    # Couverture et diversité si embeddings dispo
    if hasattr(model, 'item_embeddings'):
        res['coverage@10'] = calculate_coverage(model, test_df, n_users, n_items, user_features, item_features, 10)
        res['diversity@10'] = calculate_diversity(model, test_df, n_users, n_items, user_features, item_features, 10)
    return res


def calculate_coverage(model, test_df, n_users, n_items, user_features=None, item_features=None, k=10):
    """
    Pourcentage d'items recommandés parmi tous.
    Retourne un float.
    """
    users = test_df['user_idx'].unique()
    if users.size > 200:
        users = np.random.choice(users, 200, replace=False)
    recs = set()
    for u in users:
        scores = model.predict(u, np.arange(n_items), user_features=user_features, item_features=item_features)
        topk = np.argsort(-scores)[:k]
        recs |= set(topk.tolist())
    return len(recs) / n_items


def calculate_diversity(model, test_df, n_users, n_items, user_features=None, item_features=None, k=10):
    """
    Distance moyenne pairwise entre recommandations.
    Retourne un float.
    """
    if not hasattr(model, 'item_embeddings'):
        return 0.0
    emb = model.item_embeddings
    users = test_df['user_idx'].unique()
    users = np.random.choice(users, min(100, len(users)), replace=False)
    divs = []
    for u in users:
        scores = model.predict(u, np.arange(min(n_items, emb.shape[0])), user_features=user_features, item_features=item_features)
        topk = np.argsort(-scores)[:k]
        if topk.size < 2:
            continue
        sims = []
        for i in range(len(topk)):
            for j in range(i+1, len(topk)):
                vi, vj = emb[topk[i]], emb[topk[j]]
                sim = np.dot(vi, vj) / (np.linalg.norm(vi) * np.linalg.norm(vj) + 1e-9)
                sims.append(1 - sim)
        divs.append(np.mean(sims))
    return np.mean(divs) if divs else 0.0


def plot_learning_curves(train_metrics, test_metrics, metric_names, epochs, model_name):
    """
    Trace les courbes apprentissage et sauvegarde en PNG.
    """
    for met in metric_names:
        plt.figure()
        plt.plot(epochs, [m[met] for m in train_metrics], label='train')
        plt.plot(epochs, [m[met] for m in test_metrics], label='test')
        plt.title(f"{met} - {model_name}")
        plt.xlabel('Epochs')
        plt.ylabel(met)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{model_name}_{met}.png")
        plt.close()
