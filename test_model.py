import pandas as pd
import numpy as np
import scipy.sparse as sp
import os, sys
from functools import partial

root = os.path.dirname(os.path.realpath(__file__))
sys.path.append(root+'/RecSys2019_DeepLearning_Evaluation')

from Conferences.RecSys.SpectralCF_our_interface.Movielens1M.Movielens1MReader import Movielens1MReader
from Conferences.RecSys.SpectralCF_our_interface.MovielensHetrec2011.MovielensHetrec2011Reader import MovielensHetrec2011Reader
from Conferences.RecSys.SpectralCF_our_interface.AmazonInstantVideo.AmazonInstantVideoReader import AmazonInstantVideoReader
from Conferences.KDD.CollaborativeVAE_our_interface.Citeulike.CiteulikeReader import CiteulikeReader
from Conferences.SIGIR.CMN_our_interface.Pinterest.PinterestICCVReader import PinterestICCVReader
from Conferences.SIGIR.CMN_our_interface.Epinions.EpinionsReader import EpinionsReader
from Conferences.IJCAI.ConvNCF_our_interface.YelpReader.YelpReader import YelpReader

from Utils.ResultFolderLoader import ResultFolderLoader, generate_latex_hyperparameters
from Utils.assertions_on_data_for_experiments import assert_implicit_data, assert_disjoint_matrices
from Utils.plot_popularity import plot_popularity_bias, save_popularity_statistics

from Base.Evaluation.Evaluator import Evaluator, EvaluatorHoldout, EvaluatorNegativeItemSample
from Base.DataIO import DataIO



from ParameterTuning.SearchBayesianSkopt import SearchBayesianSkopt
from ParameterTuning.SearchAbstractClass import SearchInputRecommenderArgs
from skopt.space import Real, Integer, Categorical
import pickle

from Base.BaseSimilarityMatrixRecommender import BaseSimilarityMatrixRecommender
from scipy.stats import beta

METHOD = "crediblendcg"

hyperparameters_range_dictionary = {}
hyperparameters_range_dictionary["topK"] = Integer(1, 5000)
hyperparameters_range_dictionary["b"] = Integer(2, 300)
hyperparameters_range_dictionary["bb"] = Integer(2, 300)
hyperparameters_range_dictionary["popcoef"] = Real(0, 8)
hyperparameters_range_dictionary["recomb"] = Real(0, 1)
hyperparameters_range_dictionary["percentile"] = Real(0.00001, 0.1)

def mle(a, b, r, y):
    return ((a*(y - 1) + a*y + b*y + r*(y - 1)) - np.sqrt((-a*(y - 1) - a*y - b*y - r*(y - 1))**2 - 4*a*y*(a*(y - 1) + b*(y - 1) + r*(y - 1))))/(2*(a*(y - 1) + b*(y - 1) + r*(y - 1)))

def tosparsediag(vec):
    sd = sp.lil_matrix((vec.shape[0],vec.shape[0]))
    sd.setdiag(vec)
    return sd.tocsr()

def mle_sparse(a, bpr, sr, y):
    brym1 = sr.copy().tocoo()
    brym1.data *= -1
    brym1.data += \
        bpr[brym1.nonzero()[0], 0] * y[0, brym1.nonzero()[1]] \
        + (a*(y-1)+a*y)[0, brym1.nonzero()[1]]
    minusb = brym1.tocsr()


    aa = sr.copy().tocoo()
    aa.data = (y-1)[0, aa.nonzero()[1]] * bpr[aa.nonzero()[0], 0]
    aa.data += (a*(y - 1))[0, aa.nonzero()[1]]
    aa_csr = aa.tocsr()

    aac4 = aa.copy()
    aac4.data *= 4 * a * y[0, aac4.nonzero()[1]]

    frac = (minusb - (minusb.power(2) - aac4).sqrt()).tocoo()

    frac.data /= np.array(aa_csr[frac.nonzero()]).squeeze()*2
    return frac.toarray().squeeze()

class Knn(BaseSimilarityMatrixRecommender):
    RECOMMENDER_NAME = "Knn"

    def __init__(self, URM_train):
        super().__init__(URM_train)
        URM_train_orig = self.URM_train.copy()
        URM_train = URM_train_orig.astype(np.float32).tocsr().copy()
        self.URM_train = URM_train

    def fit(self, topK=50, percentile=0.05, bb=100, a=1.01, b=20, recomb=0, popcoef=5):
        self.fit_rowcol_credible(topK, a=1.01, b=bb, percentile=percentile)
        self.fit_weights(a, b, recomb, popcoef)

    def fit_rowcol_credible(self, topK=50, a=1.01, b=100, percentile=0.05, run_progress=True):
        blim = np.bincount((self.URM_train).tocsr().nonzero()[1]).max()+b
        alim = blim
        lookup = np.zeros((alim+1, blim+1))
        args = np.array([(a,b) for a in range(1, alim+1) for b in range(1, blim+1)])
        pweights = beta.ppf(percentile, args[:,0]+a, args[:,1])
        lookup[tuple(zip(*[(a,b) for a in range(1, alim+1) for b in range(1, blim+1)]))] = pweights

        block_size = 2000
        URM_train = self.URM_train.copy()

        all_cols = np.arange(0, URM_train.shape[1], block_size)
        all_ranges = list(zip(all_cols, list(all_cols[1:])+[URM_train.shape[1]]))
        
        dataMatrix = URM_train.tocsc()
        sumOfSquared = np.array(dataMatrix.sum(axis=0)).ravel()
        n_rows, n_columns = dataMatrix.shape
        
        rows = []
        cols = []
        values = []
    
        numrange = len(all_ranges)
        percent_lim = max(numrange // 100, 1)
        for j, (block_start, block_end) in enumerate(all_ranges):
            if j%percent_lim == 0:
                print(j, '/', numrange, end=', ', flush=True) if run_progress else None
            common_views_sparse = URM_train.T.dot(URM_train[:, block_start:block_end])
            common_views = common_views_sparse.toarray().squeeze()
            pweights = lookup[((common_views).T.astype(int), (-common_views+sumOfSquared.reshape(-1, 1)+b).T.astype(int))]
            pweights[:, block_start:block_end] *= (-np.eye(block_end-block_start)+1)
            relevant_partition = (-pweights).argpartition(axis=1, kth=topK-1)[:, 0:topK]
            
            for c in range(URM_train[:, block_start:block_end].shape[1]):
                #self.c = c
                this_column_weights = pweights[c]
                
                #this_column_weights[block_start+c] = 0.0

                #relevant_items_partition = (-this_column_weights).argpartition(topK-1)[0:topK]
                #relevant_items_partition_sorting = np.argsort(-this_column_weights[relevant_items_partition])
                #top_k_idx = relevant_items_partition[relevant_items_partition_sorting]
                top_k_idx = relevant_partition[c]

                notZerosMask = this_column_weights[top_k_idx] != 0.0

                rows.append(top_k_idx[notZerosMask])
                cols.append(c + block_start)
                weights = this_column_weights[top_k_idx][notZerosMask]
                values.append(weights)


        self.rows = rows
        self.cols = cols
        self.values = values

    def fit_weights(self, a=2, b=20, recomb=0, popcoef=5, run_progress=True):
        block_size = 2000

        self.recomb = recomb
        
        ut = np.array(self.URM_train.nonzero()).T
        item_counts = np.zeros(self.URM_train.shape[1])
        item_counts_ = np.bincount(ut[:,1])
        item_counts[:item_counts_.shape[0]] = item_counts_
        user_num = self.URM_train.shape[0]
        self.pop = item_counts*popcoef/user_num
        pop = self.pop
        
        all_cols = np.arange(0, self.URM_train.shape[1], block_size)
        all_ranges = list(zip(all_cols, list(all_cols[1:])+[self.URM_train.shape[1]]))
        axis0sum = self.URM_train.sum(axis=0)

        values = []
        cols = []
        rows = []
        neighbors = dict(zip(self.cols, self.rows))

        numrange = len(all_ranges)
        percent_lim = max(numrange // 100, 1)
        for j, (block_start, block_end) in enumerate(all_ranges):
            if j%percent_lim == 0:
                print(j, '/', numrange, end=', ', flush=True) if run_progress else None
            common_views_sparse = self.URM_train.T.dot(self.URM_train[:, block_start:block_end])
            conditional_probs = mle_sparse(a-1, np.array(axis0sum).T+b-1, common_views_sparse, pop[block_start:block_end].reshape(1,-1))
            conditional_probs[block_start:block_end] *= (-np.eye(block_end-block_start)+1)

            for c in range(self.URM_train[:, block_start:block_end].shape[1]):
                if c + block_start not in neighbors:
                    continue
                conditional_probs_col = conditional_probs[:, c]
                top_k_idx = neighbors[c + block_start]
                weights = conditional_probs_col[top_k_idx]
                rows.extend(top_k_idx)
                cols.extend(np.ones(len(top_k_idx)) * c + block_start)
                values.extend(weights)
        
        self.W_sparse = sp.csr_matrix(
            (values, (rows, cols)),
            shape=(self.URM_train.shape[1], self.URM_train.shape[1]),
            dtype=np.float32)
        self.W_sparse_neg = self.W_sparse.copy()
        self.W_sparse_neg.data = np.log(1-self.W_sparse_neg.data)

    def _compute_item_score(self, user_id_array, items_to_compute = None):
        user_profile_array = self.URM_train[user_id_array]
        if(self.W_sparse_neg is not False):
            success_ = 1-np.exp(user_profile_array.dot(self.W_sparse_neg).toarray().squeeze())
            r = self.recomb
            pops = self.pop.reshape(1, -1)
            scores = success_ * (1-pops)/(1-r*pops)+((1-r)*pops)/(1-r*pops)
        else:
            scores = user_profile_array.dot(self.W_sparse).toarray().squeeze()

        if(len(scores.shape) == 1):
            scores = scores.reshape(1, -1)

        if items_to_compute is not None:
            item_scores = - np.ones((len(user_id_array), self.URM_train.shape[1]), dtype=np.float32)*np.inf
            item_scores[:, items_to_compute] = scores[:, items_to_compute]
            return item_scores
        else:
            return scores


def read_amazbook():
    data = np.load('processed_data/amazon_book_positive.npy')
    pd_data = pd.DataFrame(data)
    np.random.seed(123)
    rand = np.random.rand(pd_data.shape[0])
    pd_data['set'] = 0
    pd_data.loc[(rand>0.90) & (rand <= 0.95), 'set'] = 1
    pd_data.loc[rand>0.95, 'set'] = 2

    np.random.seed(123)
    np_train = pd_data[pd_data['set'] == 0][[0,1]].values
    np_valid = pd_data[pd_data['set'] == 1][[0,1]].values
    np_test = pd_data[pd_data['set'] == 2][[0,1]].values
    data = (np_train, np_valid, np_test)

    shape = np.maximum(np_train.max(axis=0), np_test.max(axis=0), np_valid.max(axis=0))+1

    URM_train = sp.coo_matrix((np.ones(np_train.shape[0]), np_train.T), shape=tuple(shape)).tocsr()
    URM_test = sp.coo_matrix((np.ones(np_test.shape[0]), np_test.T), shape=tuple(shape)).tocsr()
    URM_validation = sp.coo_matrix((np.ones(np_valid.shape[0]), np_valid.T), shape=tuple(shape)).tocsr()

    return URM_train, URM_validation, URM_test

def read_amazvid():
    np_load_old = partial(np.load)
    np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
    os.chdir("./RecSys2019_DeepLearning_Evaluation")
    np.load = np_load_old
    result_folder_path = "result_experiments/{}_{}/".format('recsys', 'amazvid')
    dataset = AmazonInstantVideoReader(result_folder_path)
    os.chdir("..")

    URM_train = dataset.URM_DICT["URM_train"].copy()
    URM_validation = dataset.URM_DICT["URM_validation"].copy()
    URM_test = dataset.URM_DICT["URM_test"].copy()

    return URM_train, URM_validation, URM_test

def read_hetrec():
    np_load_old = partial(np.load)
    np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
    os.chdir("./RecSys2019_DeepLearning_Evaluation")
    np.load = np_load_old
    result_folder_path = "result_experiments/{}_{}/".format('recsys', 'hetrec')
    dataset = MovielensHetrec2011Reader(result_folder_path)
    os.chdir("..")

    URM_train = dataset.URM_DICT["URM_train"].copy()
    URM_validation = dataset.URM_DICT["URM_validation"].copy()
    URM_test = dataset.URM_DICT["URM_test"].copy()

    return URM_train, URM_validation, URM_test

def read_ml1m():
    np_load_old = partial(np.load)
    np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
    os.chdir("./RecSys2019_DeepLearning_Evaluation")
    np.load = np_load_old
    result_folder_path = "result_experiments/{}_{}/".format('recsys', 'ml1m')
    dataset = Movielens1MReader(result_folder_path, type ="ours", cold_start=None, cold_items=None)
    os.chdir("..")

    URM_train = dataset.URM_DICT["URM_train"].copy()
    URM_validation = dataset.URM_DICT["URM_validation"].copy()
    URM_test = dataset.URM_DICT["URM_test"].copy()

    return URM_train, URM_validation, URM_test

def read_epinions():
    np_load_old = partial(np.load)
    np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
    os.chdir("./RecSys2019_DeepLearning_Evaluation")
    np.load = np_load_old
    result_folder_path = "result_experiments/{}_{}/".format('recsys', 'epinions')
    dataset = EpinionsReader(result_folder_path)
    os.chdir("..")
    URM_train = dataset.URM_DICT["URM_train"].copy()
    URM_valid = dataset.URM_DICT["URM_validation"].copy()
    URM_test = dataset.URM_DICT["URM_test"].copy()
    return URM_train, URM_valid, URM_test

def read_pinterest():
    np_load_old = partial(np.load)
    np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
    os.chdir("./RecSys2019_DeepLearning_Evaluation")
    np.load = np_load_old
    result_folder_path = "result_experiments/{}_{}/".format('recsys', 'pinterest')

    from Conferences.SIGIR.CMN_our_interface.Pinterest.PinterestICCVReader import PinterestICCVReader
    dataset = PinterestICCVReader(result_folder_path)  
    os.chdir("..")
    URM_train = dataset.URM_DICT["URM_train"].copy()
    URM_valid = dataset.URM_DICT["URM_validation"].copy()
    URM_test = dataset.URM_DICT["URM_test"].copy()
    return URM_train, URM_valid, URM_test

datasets = {
    'amazbook': read_amazbook,
    'amazvid': read_amazvid,
    'hetrec': read_hetrec,
    'ml1m': read_ml1m,
    'epinions': read_epinions,
    'pinterest': read_pinterest,
}

def stats(URM_train, URM_valid, URM_test):
    URM = URM_train+URM_valid+URM_test
    print("shape", URM.shape)
    user_vals, item_vals = URM.nonzero()
    print("density", user_vals.shape[0]/(URM.shape[0]*URM.shape[1]))
    print("interactions", user_vals.shape[0])
    print("users", np.unique(user_vals).shape[0])
    print("items", np.unique(item_vals).shape[0])
    print("valid size", URM_valid.nonzero()[0].shape[0]/URM.nonzero()[0].shape[0])
    print("test size", URM_test.nonzero()[0].shape[0]/URM.nonzero()[0].shape[0])


# for k,v in datasets.items():
#     print(k)
#     # train, valid, test = v()
#     stats(*v())
# quit()

import pathos.multiprocessing as multiprocessing

def run_eval(k,v):

    EXPERIMENT = "%s_%s" % (k, METHOD)

    result_folder_path = "result_experiments/{}_{}/".format('recsys', k)

    URM_train, URM_validation, URM_test = v()

    if hyperparameters_range_dictionary["topK"].high > URM_train.shape[1]:
        hyperparameters_range_dictionary["topK"] = Integer(
            hyperparameters_range_dictionary["topK"].low,
            hyperparameters_range_dictionary["topK"].high
        )

    # Ensure IMPLICIT data and DISJOINT sets
    assert_implicit_data([URM_train, URM_validation, URM_test])
    assert_disjoint_matrices([URM_train, URM_validation, URM_test])


    # If directory does not exist, create
    if not os.path.exists(result_folder_path):
        os.makedirs(result_folder_path)

    algorithm_dataset_string = "{}_{}_".format("recsys", k)

    plot_popularity_bias([URM_train + URM_validation, URM_test],
                         ["Train data", "Test data"],
                         result_folder_path + algorithm_dataset_string + "popularity_plot")

    save_popularity_statistics([URM_train + URM_validation + URM_test, URM_train + URM_validation, URM_test],
                               ["URM_all", "URM train", "URM test"],
                               result_folder_path + algorithm_dataset_string + "popularity_statistics")


    from Base.Evaluation.Evaluator import EvaluatorHoldout
    cold_start = False
    cutoff_list_validation = [50]
    cutoff_list_test = [20, 30, 40, 50, 60, 70, 80, 90, 100]

    evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=cutoff_list_validation)
    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=cutoff_list_test)


    metric_to_optimize = "NDCG"

    from ParameterTuning.SearchBayesianSkopt import SearchBayesianSkopt
    from ParameterTuning.SearchAbstractClass import SearchInputRecommenderArgs
    import pickle

    parameterSearch = SearchBayesianSkopt(
        Knn,
        evaluator_validation=evaluator_validation,
        evaluator_test=evaluator_test
    )

    recommender_parameters = SearchInputRecommenderArgs(
        CONSTRUCTOR_POSITIONAL_ARGS = [URM_train],
        CONSTRUCTOR_KEYWORD_ARGS = {},
        FIT_POSITIONAL_ARGS = [],
        FIT_KEYWORD_ARGS = dict()
    )

    recommender_input_args_last_test = recommender_parameters.copy()
    recommender_input_args_last_test.CONSTRUCTOR_POSITIONAL_ARGS[0] = URM_train + URM_validation

    parameterSearch.search(
        recommender_parameters,
        parameter_search_space = hyperparameters_range_dictionary,
        n_cases = 150,
        n_random_starts = 5,
        output_folder_path = 'paramsearch_out/',
        output_file_name_root = EXPERIMENT,
        metric_to_optimize = metric_to_optimize,
        recommender_input_args_last_test = recommender_input_args_last_test
    )

    dataIO = DataIO(folder_path='./')
    d = dataIO.load_data(file_name = 'paramsearch_out/%s_metadata.zip'%EXPERIMENT)
    best = d['hyperparameters_best_index']

    print(d['hyperparameters_list'][best])
    print(d['result_on_validation_list'][best])
    print(d['result_on_test_list'][best])
    print(d['result_on_last'])

    # break

pool = multiprocessing.Pool(processes=10, maxtasksperchild=1)
res = pool.map(
    lambda r: run_eval(r[0],r[1]), datasets.items()
)
pool.close()
pool.join()
