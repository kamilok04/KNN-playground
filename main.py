import unittest
import pandas as pd
import time
from KNN import *  # adjust if class is in a subfolder or renamed

class TestKNNAccuracy(unittest.TestCase):

    def setUp(self):
        self.data_path = 'iris.csv'
        self.metrics = ['euclidean', 'manhattan', 'minkowski', 'chebyshev']
        self.k_values = [3, 5, 7]
        self.voting = ['soft', 'hard']
        self.target = ['variety']

    def test_mtree(self):
        self.driver('mtree')

    def test_bruteforce(self):
        self.driver('bruteforce')

    def driver(self, knn_method):
        start = time.time()
        results = []
        for metric in self.metrics:
            for k in self.k_values:
                for voting in self.voting:
                    model = KNN(self.data_path, k, classes_to_determine=self.target, reduce_dataset=True, knn_method=knn_method, distance_metric=metric, voting_method=voting)
                    model.distance_metric = metric  # Inject metric directly
                    model.preprocess_data()
                    control, test = model.shuffle()['normalized']
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
                    results.append((metric, k, voting, accuracy))
                    print(f"Method: {knn_method}, Metric: {metric}, k: {k}, Voting Method: {voting}, Accuracy: {accuracy:.2f}%")

                    self.assertGreater(accuracy,50, f"Low accuracy: {accuracy} with metric={metric}, k={k}")
            print("\n")   
        stop = time.time()
        print(f'Test zajął {stop - start} sekund.')

    def calculate_accuracy(self, predictions, expected):
        expected_list = expected.to_numpy().tolist()
        matches = sum(1 for pred, exp in zip(predictions, expected_list) if list(pred) == exp )
        return (matches / len(expected)) * 100


if __name__ == '__main__':
    unittest.main()
