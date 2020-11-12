import sys, os
sys.path.append('../')

from DRecPy.Recommender.Baseline import ItemKNN
from DRecPy.Dataset import get_train_dataset, get_test_dataset
from DRecPy.Evaluation.Splits import leave_k_out
from DRecPy.Evaluation.Processes import ranking_evaluation
from DRecPy.Evaluation.Metrics import Recall, ReciprocalRank
import time

ds_train = get_train_dataset('lastfm')
ds_test = get_test_dataset('lastfm')

start_train = time.time()
item_cf = ItemKNN(k=5, m=1, shrinkage=50, sim_metric='adjusted_cosine', verbose=True)
item_cf.fit(ds_train)
print("Training took", time.time() - start_train)

start_evaluation = time.time()
print(ranking_evaluation(item_cf, ds_test, n_pos_interactions=1, n_neg_interactions=5,  generate_negative_pairs=True,
                         novelty=True, k=list(range(1, 11)), metrics=[ReciprocalRank(), Recall()], seed=10))
print("Evaluation took", time.time() - start_evaluation)
