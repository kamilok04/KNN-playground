import unittest
import pandas as pd
import time
import concurrent.futures
from KNN import *  # adjust if class is in a subfolder or renamed

class TestKNNAccuracy(unittest.TestCase):

    def setUp(self):
        self.data_path = 'heart.csv'
        self.metrics = ['euclidean', 'manhattan', 'minkowski', 'chebyshev']
        self.normalizations = [ 'minmax', 'zscore', 'decimal_scaling', 'robust', 'none']
        self.k_values = [3, 5, 7]
        self.voting = ['soft', 'hard']
        self.target = ['HeartDisease']
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=14) # recommended: no. of your CPU cores + 2

    #def test_mtree(self):
    #    self.driver('mtree')

    def test_bruteforce(self):
        self.driver('bruteforce')

    def execute(self, model, normalized_data):
        control, test = model.shuffle(normalized_data)
        prepared_test = test.copy()
        prepared_test = model.remove_unknown_properties(prepared_test)
        classified_test = None
        if model.reduce_dataset:
            reduce_keep, reduce_discard = model.CNN_reduction(control)
            match model.knn_method:
                case 'mtree':
                    classified_test = model.classify_test_set(reduce_keep, prepared_test)
                case 'bruteforce':
                    for item in reduce_discard: control.drop(item.name)
                    classified_test = model.classify_test_set(control, prepared_test)
        if classified_test is None or len(classified_test) != len(test):
            raise Exception('something got derailed!')
    
        accuracy = self.calculate_accuracy(classified_test, test[self.target])
        print(f"Method: {model.knn_method}, Normalization technique: {model.normalization_method}, Metric: {model.distance_metric}, k: {model.k}, Voting Method: {model.voting_method}, Accuracy: {accuracy:.2f}%")

        self.assertGreater(accuracy,5, f"Low accuracy: {accuracy} with metric={model.distance_metric}, k={model.k}")

    def driver(self, knn_method):
        start = time.time()
        results = []
        model = KNN(self.data_path, 0, classes_to_determine=self.target, knn_method = knn_method)
        model.preprocess_data()
        for normalization in self.normalizations:
            model.normalization_method = normalization
            normalized_data = model.normalize_dataset(model.df, model.normalization_method)
            for metric in self.metrics:
                model.distance_metric = metric
                for k in self.k_values:
                    model.k = k
                    for voting in self.voting:
                        model.voting_method = voting
                        self.pool.submit(self.execute(model, normalized_data))
                                               
            print("\n")   
        stop = time.time()
        print(f'Test zajął {stop - start} sekund.')

    def calculate_accuracy(self, predictions, expected):
        
        expected_list = expected.to_numpy().tolist()
        matches = sum(1 for pred, exp in zip(predictions, expected_list) if list(pred) == exp )
        return (matches / len(expected)) * 100


if __name__ == '__main__':
    unittest.main()
