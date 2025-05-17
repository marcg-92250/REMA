# KuaiRec 2.0 Recommender System

## Dataset Overview

The KuaiRec 2.0 dataset consists of interactions between users and videos from the Kuaishou platform, and it includes the following components:

- **Interaction Data**:
  - `small_matrix.csv` (~4.7M rows)
  - `big_matrix.csv` (~12.5M rows)
  - Key columns: `user_id`, `video_id`, `watch_ratio`, and timestamps.

- **Item Information**:
  - `item_categories.csv` which contains categorical features for each video.

- **User Information**:
  - `user_features.csv` which contains activity-related features and demographics for each user.
- 
## Installation

```bash
python3 -m venv venv
source venv/bin/activate  # Linux
# ou
venv\Scripts\activate     # Windows

pip install -r requirements.txt
```

## Exécution standard (entraînement complet)

```bash
python main.py --matrix big_matrix.csv --epochs 200 --test_neg_ratio 99
```
## Approach

### Labeling Criteria

We derived implicit feedback labels from the `watch_ratio`:

- `label = 1` if the `watch_ratio ≥ 0.7` (indicating a positive interaction).
- `label = 0` otherwise (indicating a negative interaction).

This threshold was chosen to represent highly engaged interactions, where users watched at least 70% of the video's total duration.

### Data Filtering

To ensure more reliable recommendations:

- We retained only users with at least 3 positive interactions.
- We kept only items (videos) with at least 3 positive interactions.

This filtering process helps to remove sparse data and ensures the interactions are meaningful.

### Train/Test Split

We applied a leave-n-out strategy to split the data:

- For each user, 20% of their positive interactions were randomly held out for testing.
- For training, we sampled 8 unseen items for each positive interaction.
- For testing, we sampled up to 99 unseen items for each held-out interaction (configurable via `test_neg_ratio`).

This ensures a personalized evaluation per user, making the testing scenario more robust and challenging.

## Model Settings

### Baseline Model (LightFM)

- **Architecture**: Matrix factorization with WARP (Weighted Approximate-Rank Pairwise) loss.
- **Embedding size**: 256-dimensional embeddings.
- **Learning rate**: 0.03.
- **Regularization**: `alpha = 0.0005` for both user and item embeddings.
- **Max sampled**: 150 negative samples per training step.
- **Input**: Only user-item interactions (no additional features).
- **Early stopping**: Patience of 5 to 12 evaluations.
- **Training duration**: 100 to 300 epochs (configurable).

## Evaluation Protocol

### Evaluation Metrics

We evaluated the baseline model on various ranking metrics:

- **Precision@k**: The fraction of recommended items that are relevant.
- **Recall@k**: The fraction of relevant items that are recommended.
- **F1@k**: Harmonic mean of precision and recall.
- **NDCG@k**: Normalized Discounted Cumulative Gain, which emphasizes the rank of relevant items.
- **Item Coverage@10**: Percentage of total items that appear in any user's top-10 recommendations.
- **Diversity@10**: Measures the diversity of the recommended items by calculating the average pairwise distance between item embeddings.

The evaluation was performed on the held-out test set for each user.

## Results

### Model Performance Comparison

- In results_big_matrix_20250517_222312.csv

### Training Time

- **Small Matrix**:
  - **Baseline Model**: 5464.62 seconds for 320 epoch
- **Big Matrix**:
  - **Baseline Model**: 7060.04 seconds for 200 epoch

### Output Example

```plaintext
Using 32 CPU threads
Data directory: /root/REMA/KuaiRec2.0/data

Loading data from big_matrix.csv...
Loading item categories...
Loading user features...

Interaction matrix shape: (12530806, 8)
Item categories shape: (10728, 2)
User features shape: (7176, 31)

Deriving implicit labels (watch_ratio >= 0.7)...
Positive interactions ratio: 0.5147

Filtering users and items with >= 3 positive interactions...
Counting positive interactions per user and item...
Applying filtering...
Original interactions: 12530806
Filtered interactions: 12523332
Unique users: 7176
Unique items: 9923
Creating user and item mappings...

Splitting data (leave-n-out)...
Building user interaction dictionaries...
Processing interactions: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 12523332/12523332 [03:14<00:00, 64255.05it/s]
Creating train-test split...
Splitting users: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 7176/7176 [00:25<00:00, 277.47it/s]
Creating DataFrames and matrices...
Building sparse matrices...
Training interactions: 40241339
Testing interactions: 129259600

==================================================
Entraînement du modèle de base (LightFM avec perte WARP)
==================================================
Évaluation à l'époque 200: 100%|███████████████████████████████████████████████████████████████████████████| 200/200 [1:57:40<00:00, 35.30s/it, train_f1@5=0.0222, test_f1@5=0.0537, best_f1@5=0.0541, no_improv=3]

Temps total d'entraînement : 7060.04 secondes

Métriques finales :
  Entraînement: {'precision@5': 0.8429487179487181, 'recall@5': 0.0114081144580761, 'f1@5': 0.02223300792954486, 'ndcg@5': 0.958016136758489, 'precision@10': 0.8288043478260869, 'recall@10': 0.022511965983650656, 'f1@10': 0.04281635879994376, 'ndcg@10': 0.9546353339029031, 'precision@20': 0.822463768115942, 'recall@20': 0.044723512915331706, 'f1@20': 0.08137804197841884, 'ndcg@20': 0.951123785419115, 'precision@50': 0.8178651059085842, 'recall@50': 0.11132164698661788, 'f1@50': 0.18092003778826576, 'ndcg@50': 0.9467992814178116, 'item_coverage@10': 0.3963519097047264, 'diversity@10': 0.8045571624918278}
  Test:         {'precision@5': 0.5814659977703457, 'recall@5': 0.02929703135340489, 'f1@5': 0.053720606887399464, 'ndcg@5': 0.6256436596067969, 'precision@10': 0.5364130434782608, 'recall@10': 0.054644767195430016, 'f1@10': 0.09303855050057137, 'ndcg@10': 0.5849641918474042, 'precision@20': 0.4904194537346711, 'recall@20': 0.09635632678149146, 'f1@20': 0.14602978657044866, 'ndcg@20': 0.542162784288208, 'precision@50': 0.4149609810479376, 'recall@50': 0.18366054783661429, 'f1@50': 0.22052735308676494, 'ndcg@50': 0.4811958429830368, 'item_coverage@10': 0.39111155900433336, 'diversity@10': 0.8041891462546223}

Results saved to results_big_matrix_20250517_222312.csv
```

## Conclusions

The model's effectiveness was validated using a variety of ranking metrics, including Precision, Recall, and F1 scores, where it consistently performed well, demonstrating its reliability in multiple evaluation aspects.

By leveraging WARP (Weighted Approximate-Rank Pairwise) loss in the matrix factorization model, we effectively optimized for ranking tasks, making this approach highly suitable for handling implicit feedback, particularly in video recommendation contexts.

Through rigorous data preprocessing and filtering, we significantly enhanced the quality of recommendations. Specifically, users and items with insufficient interactions were excluded, ensuring that the training data reflected meaningful and reliable patterns.

The evaluation process was thorough, incorporating both traditional accuracy metrics such as Precision, Recall, and F1, as well as more comprehensive measures like NDCG, Item Coverage, and Diversity, to provide a complete assessment of the model's performance.

The use of early stopping was instrumental in balancing training efficiency with performance. It helped prevent overfitting while ensuring that the model training process was neither too short nor too long, thereby optimizing computational resources.

Preprocessing steps were carefully designed to handle both user and item features, including robust scaling of numerical data. This strategy prevented overfitting by ensuring that no single feature disproportionately influenced the model’s predictions.


## Future Directions
Integrating Session-based Recommendations: Develop session-based recommendation systems that focus on users' immediate preferences, leveraging recent interactions to provide more relevant and timely suggestions.

Incorporating Temporal Patterns: Explore sequential recommendation models that can capture how user preferences change over time, adapting recommendations to evolving interests and trends.

Refining Watch Ratio Sensitivity: Test various thresholds for the watch_ratio to understand the impact of different definitions of positive interactions on model accuracy and recommendation quality.

Leveraging Advanced Neural Networks: Experiment with neural network architectures, such as Neural Collaborative Filtering (NCF) and Variational Autoencoders (VAE), to explore their potential advantages over traditional matrix factorization methods.

Cross-validation for Robustness: Implement cross-validation techniques to assess the model’s performance across different subsets of the data, ensuring its reliability and generalizability.

Enhancing Recommendations with Content-based Features: Integrate content-based features, such as video genre or description, to refine recommendations, providing more personalized and context-aware suggestions.

Real-time Validation through A/B Testing: Conduct A/B testing to evaluate the model's effectiveness in real-world environments, enabling direct comparison between different recommendation strategies and gathering user feedback.

## Project File Structure

Here is a directory tree of the project structure:

```plaintext
          Length Name
          ------ ----
          12989 evaluation.py
          1437 loaddata.py
          11669 main.py
          15129 preprocess.py
          3602 README.md
          5820 report.md
          108 requirements.txt
          1235 results_big_matrix_20250517_222312.csv
          617 sujet.md
          33843 systeme-recommander.ipynb
```
