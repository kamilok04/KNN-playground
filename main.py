import pandas
import math
import heapq

# locals
import MTree
from utility import *

class KNN:
 
    def __init__(self, path, k, classes_to_determine = None, reduce_dataset = True, knn_method = 'mtree', distance_metric = 'euclidean'):
        self.df = pandas.read_csv(path)
        self.df = self.df.dropna()
        self.reduce_dataset = reduce_dataset
        self.knn_method = knn_method
        self.k = k
        if classes_to_determine is None:
            self.classes_to_determine = ['variety']
        else: self.classes_to_determine = classes_to_determine
        self.keep_mask = []
        for column in self.df:
            if column in self.classes_to_determine:
                self.keep_mask.append(False)
            else:
                self.keep_mask.append(True)
        self.distance_metric = distance_metric


    def preprocess_data(self):
        """
        Define dictionaries containing all non-number features so that they can be mapped to numbers.
        """
        for column in self.df:
            self.df = self.df.apply(lambda x: pandas.factorize(x)[0])
        

    
    def shuffle(self, data = None, ratio = 70):
        if data is None:
            data = self.df
        border = len(data) * ratio // 100
        shuffled = data.sample(frac=1)
        control = shuffled.iloc[:border]
        test = shuffled.iloc[border:]
        return control, test
        
    def CNN_reduction(self, control):
        """
        Hart, Peter E. (1968). "The Condensed Nearest Neighbor Rule". IEEE Transactions on Information Theory. 18: 515–516. doi:10.1109/TIT.1968.1054155.
        The tree created during this routine can easily be used for actual KNN, no need to regenerate
        """
        keep = []
        discard = []
        # Implement the CNN reduction algorithm
        for index, row in control.iterrows():
            if keep == []:
                keep = MTree.MTree(KNN_size=self.k)
                keep.insert(row, keep.root)
            else:
                neighbours = keep.KNN_search(row, self.k)
                classification = self.cast_votes(neighbours)
                t = tuple(row[self.classes_to_determine].to_numpy().tolist())
                if classification != t:
                    keep.insert(row, keep.root)
                else:
                    discard.append(row)
        
        changes_made = True
        while changes_made and len(discard):
            changes_made = False
            for i in range(len(discard)):
                neighbours = keep.KNN_search(discard[i], self.k)
                classification = self.cast_votes(neighbours)
                if classification != discard[i].iloc[-1]:
                    keep.insert(discard[i], keep.root)
                    discard.pop(i)
                    changes_made = True
                    break

        # print(f'Odrzuconych indeksów: {len(discard)}')
        return keep, discard

    def classify_point(self, point, control):
        match self.knn_method:
            case 'mtree':
                neighbours = control.KNN_search(point, self.k)
            case 'bruteforce':
                neighbours = []
                for _,e in control.iterrows():
                    neighbours.append((compute_distance(e, point, self.distance_metric), e))

                neighbours = sorted(neighbours,key=lambda e: e[0])[:self.k]
                neighbours = [a[1] for a in neighbours]
            case _:
                raise Exception('Invalid KNN method!')
        # print(f'Neighbours: {neighbours}'
        classification = self.cast_votes(neighbours)
        return classification

    def cast_votes(self, neighbours):
        """
        Cast votes based on the k nearest neighbors.
        Return the class with the most votes.

        TODO: soft sets go here!
        """
        votes = {}
        for neighbor in neighbours:
            classification = tuple(neighbor[self.classes_to_determine].to_numpy().tolist())
            if classification in votes:
                votes[classification] += 1
            else:
                votes[classification] = 1
        
        return max(votes, key=votes.get)

    def classify_test_set(self, control, test):
        classifications = []
        for _, test_point in test.iterrows():
            classification = self.classify_point(test_point, control)
            classifications.append(classification)
        return classifications


    def driver(self):
        self.preprocess_data()
        control, test = self.shuffle()
        prepared_test = test.copy()
        prepared_test = self.remove_unknown_properties(prepared_test)
        classified_test = None
        if self.reduce_dataset:
            reduce_keep, reduce_discard = self.CNN_reduction(control)
            match self.knn_method:
                case 'mtree':
                    classified_test = self.classify_test_set(reduce_keep, prepared_test)
                case 'bruteforce':
                    for item in reduce_discard: control.drop(item.name)
                    classified_test = self.classify_test_set(control, prepared_test)
        if classified_test is None or len(classified_test) != len(test):
            raise Exception('something got derailed!')
            return
        self.analyze_results(control, test, classified_test)

    def analyze_results(self, control, test, classifications):
        score = 0
        for actual, expected in zip(test[self.classes_to_determine].iterrows(), classifications):
            actual = tuple(actual[1].to_numpy().tolist())
            if actual == expected: score += 1
        # for actual, expected in zip(test[self.classes_to_determine]
        accuracy = score * 100 / len(test)
        print(f'Method: {self.knn_method}, metric: {self.distance_metric}, accuracy: {accuracy}%')

        

        
    def remove_unknown_properties(self, dataset):
        return dataset[dataset.columns[self.keep_mask]]
        

knn = KNN('iris.csv', 1, knn_method='bruteforce')
for metric in ('euclidean', 'manhattan', 'minkowski'):
    knn.distance_metric = metric
    for method in ('mtree', 'bruteforce'):
        knn.knn_method = method
        knn.driver()