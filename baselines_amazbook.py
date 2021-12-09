import pandas as pd
import numpy as np
import scipy.sparse as sp
import os, sys

root = os.path.dirname(os.path.realpath(__file__))
sys.path.append(root+'/RecSys2019_DeepLearning_Evaluation')

from Base.Evaluation.Evaluator import Evaluator, EvaluatorHoldout

from Recommender_import_list import *

from ParameterTuning.run_parameter_search import runParameterSearch_Collaborative

import os, traceback, argparse
from functools import partial
import numpy as np

from Base.DataIO import DataIO

from ParameterTuning.SearchSingleCase import SearchSingleCase
from ParameterTuning.SearchAbstractClass import SearchInputRecommenderArgs

from Utils.ResultFolderLoader import ResultFolderLoader, generate_latex_hyperparameters
from Utils.assertions_on_data_for_experiments import assert_implicit_data, assert_disjoint_matrices
from Utils.plot_popularity import plot_popularity_bias, save_popularity_statistics


def read_data_split_and_search(dataset_name,
                                   flag_baselines_tune = False,
                                   flag_DL_article_default = False, flag_DL_tune = False,
                                   flag_print_results = False):

    result_folder_path = "result_experiments/{}_{}/".format('recsys', 'amazbook')


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
    URM_valid = sp.coo_matrix((np.ones(np_valid.shape[0]), np_valid.T), shape=tuple(shape)).tocsr()

    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[20])
    evaluator_valid = EvaluatorHoldout(URM_valid, cutoff_list=[20])

    cutoff_list_validation = [50]
    cutoff_list_test = [20, 30, 40, 50, 60, 70, 80, 90, 100]

    evaluator_valid = EvaluatorHoldout(URM_valid, cutoff_list=cutoff_list_validation)
    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=cutoff_list_test)

    # If directory does not exist, create
    if not os.path.exists(result_folder_path):
        os.makedirs(result_folder_path)


    collaborative_algorithm_list = [
        Random,
        TopPop,
        UserKNNCFRecommender,
        ItemKNNCFRecommender,
        P3alphaRecommender,
        RP3betaRecommender,
        # PureSVDRecommender,
        # NMFRecommender,
        # IALSRecommender,
        # MatrixFactorization_BPR_Cython,
        # MatrixFactorization_FunkSVD_Cython,
        EASE_R_Recommender,
        # SLIM_BPR_Cython,
        SLIMElasticNetRecommender,
    ]

    metric_to_optimize = "NDCG"


    algorithm_dataset_string = "{}_{}_".format(ALGORITHM_NAME, dataset_name)

    plot_popularity_bias([URM_train + URM_valid, URM_test],
                         ["URM train", "URM test"],
                         result_folder_path + algorithm_dataset_string + "popularity_plot")

    save_popularity_statistics([URM_train + URM_valid, URM_test],
                               ["URM train", "URM test"],
                               result_folder_path + algorithm_dataset_string + "popularity_statistics")



    from Base.Evaluation.Evaluator import EvaluatorNegativeItemSample

    runParameterSearch_Collaborative_partial = partial(runParameterSearch_Collaborative,
                                                       URM_train = URM_train,
                                                       URM_train_last_test = URM_train + URM_valid,
                                                       metric_to_optimize = metric_to_optimize,
                                                       evaluator_validation_earlystopping = evaluator_valid,
                                                       evaluator_validation = evaluator_valid,
                                                       evaluator_test = evaluator_test,
                                                       output_folder_path = result_folder_path,
                                                       parallelizeKNN = False,
                                                       allow_weighting = True,
                                                       resume_from_saved = True,
                                                       n_cases = 150,
                                                       n_random_starts = 5)




    if flag_baselines_tune:

        for recommender_class in collaborative_algorithm_list:
            try:
                runParameterSearch_Collaborative_partial(recommender_class)
            except Exception as e:
                print("On recommender {} Exception {}".format(recommender_class, str(e)))
                traceback.print_exc()






if __name__ == '__main__':

    ALGORITHM_NAME = "AMAZBOOK"
    CONFERENCE_NAME = "RECSYS"

    KNN_similarity_to_report_list = ["cosine", "dice", "jaccard", "asymmetric", "tversky"]

    dataset_name="asd"
    read_data_split_and_search(dataset_name, flag_baselines_tune=True)


