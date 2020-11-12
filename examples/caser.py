import sys, os
sys.path.append('../')

from DRecPy.Recommender import Caser
from DRecPy.Dataset import get_train_dataset, get_test_dataset
from DRecPy.Evaluation.Splits import leave_k_out
from DRecPy.Evaluation.Processes import recommendation_evaluation, ranking_evaluation
from DRecPy.Evaluation.Metrics import Recall, ReciprocalRank

ds_train = get_train_dataset('lastfm')
ds_test = get_test_dataset('lastfm')

caser = Caser(L=5, T=3, d=50, n_v=4, n_h=16, dropout_rate=0.5, sort_column='timestamp', seed=42)
caser.fit(ds_train, epochs=350, batch_size=2 ** 12, learning_rate=0.005, reg_rate=1e-6, neg_ratio=3)


print(recommendation_evaluation(caser, ds_test, novelty=True, k=[10],
                                metrics=[Recall()], seed=42,
                                max_concurrent_threads=12))

print(ranking_evaluation(caser, ds_test, n_pos_interactions=1, n_neg_interactions=5,  generate_negative_pairs=True,
                         novelty=True, k=[10], metrics=[ReciprocalRank()], seed=10))

# 'AveragePrecision@1': 0.232, 'AveragePrecision@5': 0.1378, 'AveragePrecision@10': 0.1123,
# 'Precision@1': 0.232, 'Precision@5': 0.2088, 'Precision@10': 0.1899,
# 'Recall@1': 0.0138, 'Recall@5': 0.062, 'Recall@10': 0.1085
