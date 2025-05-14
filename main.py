import unittest
import pandas as pd
import time
import matplotlib.pyplot as pyplot
import concurrent.futures
import logging
from KNN import *  # adjust if class is in a subfolder or renamed


class TestKNNAccuracy(unittest.TestCase):

    def setUp(self):
        self.data_path = 'heart.csv'
        self.metrics = ['euclidean', 'manhattan', 'minkowski', 'chebyshev']
        
        #self.normalizations = ['none', 'decimal_scaling', 'minmax', 'zscore',  'robust']
        
        self.normalizations = ['none']
        self.knn_methods = ['mtree', 'bruteforce']
        self.k_values = [1, 3, 5, 10]
        self.ratios = [*range(10, 100, 10)]
    
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(filename='out/log.txt', level=logging.INFO, format='%(asctime)s %(message)s', filemode='a')
        self.voting = ['soft', 'hard']
        self.target = ['HeartDisease']
        
    # Obsolete, as we pit bruteforce against m-treee in a single graph
    # def test_mtree(self): self.driver('mtree')
    # def test_bruteforce(self): self.driver('bruteforce')

    def execute(self, model, normalized_data, ratio):
        start = time.time()
        
        control, test = model.shuffle(normalized_data, ratio, keep_order = True)
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
        else:
            # is this tested?
            match model.knn_method:
                case 'mtree':
                    tree = MTree.MTree(model.df.shape, model.k, model.distance_metric)
                    for point in control: tree.append(point, tree.root)
                    classified_test = model.classify_test_set(tree, prepared_test)
                case 'bruteforce':
                    classified_test = model.classify_test_set(control, prepared_test)
        if classified_test is None or len(classified_test) != len(test):
            raise Exception('something got derailed!')
    
        matrix = self.generate_confusion_matrix(classified_test, test[self.target], model.target_entries)
        return [ratio, matrix, start]
        # TODO: remove
        # accuracy = (sum(matrix[i][i] for i in range(len(model.target_entries))) / sum(sum(row) for row in matrix)) * 100
        # print(f"Method: {model.knn_method}, Normalization technique: {model.normalization_method}, Metric: {model.distance_metric}, k: {model.k}, Voting Method: {model.voting_method}, Accuracy: {accuracy:.2f}%")



    def test_KNN(self):
        start = time.time()
        model = KNN(self.data_path, 0, classes_to_determine=self.target)
        model.preprocess_data()
        model.df = model.shuffle(keep_order = False, control_percentage = None) # shuffle once for all subsequent tests
        for normalization in self.normalizations:
            model.normalization_method = normalization
            normalized_data = model.normalize_dataset(model.df, model.normalization_method)
            for metric in self.metrics:
                model.distance_metric = metric
                for k in self.k_values:
                    model.k = k
                    for voting in self.voting:
                        model.voting_method = voting
                        matrices = {}
                        for method in self.knn_methods:
                            model.knn_method = method
                            matrices[method] = []
                            with concurrent.futures.ThreadPoolExecutor() as pool:
                                future_to_result = {pool.submit(self.execute,model, normalized_data, ratio): ratio
                                for ratio in self.ratios}
                                for future in concurrent.futures.as_completed(future_to_result):
                                    result = future.result()
                                    result[2] -= time.time()
                                    result[2] *= -1 
                                    matrices[method].append(result)
                                pool.shutdown(wait= True)
                            

                        self.generate_graphs(model, matrices)
                                               
            print("\n")   
        stop = time.time()
        print(f'Test zajął {stop - start} sekund.')
        
    def generate_graphs(self, model, matrices):
        fig, (ax1, ax2) = pyplot.subplots(2, 1)
        fig.set_size_inches(9, 10)
        fig.suptitle(f'normalization: {model.normalization_method}, distance metric: {model.distance_metric}, k = {model.k}, voting method: {model.voting_method}')
        ax1.set_title('Accuracy vs Control Ratio with different k-NN algorithms')
        ax2.set_title('Time spent computing vs Control Ratio with different k-NN algorithms')
        ax1.set_xlabel('Control Ratio (%)')
        ax1.set_ylabel('Accuracy (%)')
        ax2.set_xlabel('Control Ratio (%)')
        ax2.set_ylabel('Time spent calculating (s)')
        ax1.set_xticks(self.ratios)
        ax2.set_xticks(self.ratios)
        pyplot.xticks(self.ratios)
        legend = []
        accuracies = []
        width = 3
        offset = - width / 2
        for method, value in matrices.items():
            
            legend.append(method)
            ratios = []
            accuracies = []
            timespans = []
            for ratio, matrix, time_spent in value:
  
                correct = sum(matrix[i][i] for i in range(len(model.target_entries)))
                total = sum(sum(row) for row in matrix)
                accuracy = (correct / total) * 100
                accuracies.append(accuracy)
                timespans.append(time_spent)
                ratios.append(ratio + offset)
                logging.info(f"Method: {method}, Normalization technique: {model.normalization_method}, Metric: {model.distance_metric}, k: {model.k}, Voting Method: {model.voting_method}, Control Ratio: {ratio}% Accuracy: {accuracy:.2f}%, Execution time: {time_spent}")
            ax1.bar(ratios, accuracies, width=width)
            ax2.scatter(self.ratios, timespans)
            offset += width
          
            
        
        ax1.legend(legend)
        ax2.legend(legend)
        
        pyplot.xticks(self.ratios)
        # Save the plot with a descriptive filename
        filename = f"out/accuracy_{model.normalization_method}_{model.distance_metric}_k{model.k}_{model.voting_method}.png"
        fig.tight_layout()
        pyplot.savefig(filename)
        pyplot.close()
        
                

    def generate_confusion_matrix(self, predictions, expected, target_entries):
        """Create a confusion matrix.

        Args:
            predictions (list): Predictions made by the algorithm.
            expected (list): Actual values.
            target_entries (list): 
                A map of which target value corresponds to which class.
                (target classes are vectorized, too)

        Returns:
            _type_: _description_
        """
        expected_list = expected.to_numpy().tolist()
        confusion_matrix = [[0 for _ in target_entries] for _ in target_entries]

        # The confusion matrix
        # XXXXXXXX | Predicted A | Predicted B | ...
        # Actual A |               
        # Actual B |
        # ........

        # Predictions put on the main diagonal are correct.

        for pred, exp in zip(predictions, expected_list):
            confusion_matrix[exp[0]][int(pred)] += 1
        
      
        return confusion_matrix



        


if __name__ == '__main__':
    unittest.main()
