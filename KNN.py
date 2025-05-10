import pandas
import math
import heapq

# locals
import MTree
from utility import *

class KNN:
 
    def __init__(self, path, k, classes_to_determine = None, reduce_dataset = True, normalization_method='minmax', knn_method = 'mtree',distance_metric = 'euclidean',voting_method = 'hard'):
        """Initialize KNN searching class.

        Args:
            path (str): Path to the dataset file.
            k (int): How many neighbours do we take into account when voting?
            classes_to_determine (list, optional): 
                List of columns containing classes to determine. 
                These columns will be removed from testing sets.
                Defaults to None.
            reduce_dataset (bool, optional): 
                Should we perform CNN reduction before training? 
                Defaults to True.
            knn_method (str, optional): 
                The underlying structure of the algorithm. 
                Current options are: 'mtree', 'bruteforce'. 
            distance_metric (str, optional):
                What metric should the algorithm use when computing distances?
                Current options are 'euclidean', 'minkowski', 'manhattan'
                Defaults to 'euclidean'.
            voting_method (str, optional): 
                What voting method should the algorithm use? 
                Current options are 'hard' and 'soft'.
                Defaults to 'hard'.
        """
        self.df = pandas.read_csv(path)
        self.df = self.df.dropna()
        self.reduce_dataset = reduce_dataset
        self.normalization_method = normalization_method
        self.knn_method = knn_method
        self.voting_method = voting_method
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
        Map all non-number features onto sets of numbers. 
        """
        for column in self.df:
            self.df = self.df.apply(lambda x: pandas.factorize(x)[0])
        

    
    def shuffle(self, data : pandas.DataFrame = None, control_percentage : float = 70, normalize_only = True):
        """_summary_

        Args:
            data (pandas.DataFrame, optional): 
                A dataset to shuffle. 
                Defaults to None.
            control_percentage (float, optional): 
                Percentage of dataset which will go into the control (training) set. 
                Defaults to 70.
            normalize_only (bool, optional): 
            Should the function return only the normalized dataset?. 
            Defaults to True.

        Returns:
            dict: A dict of the structure
                { 
                  'regular' : (control, test),
                  'normalized' : (normalized_control, normalized_test)
                }
        """
        
        if data is None:
            data = self.df
        border = len(data) * control_percentage // 100
        shuffled = data.sample(frac=1, random_state=1) #good samples: 32, 1,2
        output = {}

        if not normalize_only:
            output['regular'] = (shuffled.iloc[:border], shuffled.iloc[border:])
    
        normalized_shuffled = self.normalize_dataset(shuffled)
        output['normalized'] = (
            normalized_shuffled.iloc[:border], 
            normalized_shuffled.iloc[border:]
        )
        return output

    def normalize_dataset(self, dataset, normalization_method = 'zscore'):
        """Normalize a dataset using one of the common techniques.

        Args:
            dataset (pandas.DataFrame): A dataset to normalize.
            normalization_method (str, optional): 
                A normalization method to use. 
                Defaults to 'minmax'.

        Returns:
            normalized_dataset (pandas.DataFrame):
                The normalized dataset.
        """
        normalized_dataset = dataset.copy()
        match normalization_method:
            case 'minmax':
                for column in normalized_dataset.columns:
                    normalized_dataset[column] /= normalized_dataset[column].abs().max()
            case 'zscore':
                for column in normalized_dataset.columns:
                    normalized_dataset[column] = (normalized_dataset[column] - normalized_dataset[column].mean()) / normalized_dataset[column].std()
            case 'decimal_scaling':
                for column in normalized_dataset.columns:
                    max_value = normalized_dataset[column].abs().max()
                    j = math.floor(math.log10(max_value)) + 1
                    normalized_dataset[column] /= 10 ** j
            case 'robust':
                for column in normalized_dataset.columns:
                    q1 = normalized_dataset[column].quantile(0.25)
                    q2 = normalized_dataset[column].quantile(0.5)
                    q3 = normalized_dataset[column].quantile(0.75)
                    iqr = q3 - q1
                    normalized_dataset[column] = (normalized_dataset[column] - q2) / iqr
            case _:
                print('Unknown normalization technique!')

        return normalized_dataset
        
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
                keep = MTree.MTree(KNN_size=self.k,distance_metric=self.distance_metric)
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

    def classify_point(self, point, control, key = 'regular'):
        match self.knn_method:
            case 'mtree':
                neighbours = control.KNN_search(point, self.k)
            case 'bruteforce':
                neighbours = []
                for _,e in control.iterrows():
                    neighbours.append((compute_distance(e, point, self.distance_metric), e))

                neighbours = sorted(neighbours,key=lambda e: e[0])[:self.k]
                neighbours = [(a[1], a[0]) for a in neighbours]
            case _:
                raise Exception('Invalid KNN method!')
        # print(f'Neighbours: {neighbours}'
        classification = self.cast_votes(neighbours)
        return classification

    def cast_votes(self, neighbours):
        """
        Cast votes based on the k nearest neighbors.
        Return the class with the most votes.
        """
        votes = {}
        if self.voting_method == 'hard':
            for neighbor, distance in neighbours:
                classification = tuple(neighbor[self.classes_to_determine].to_numpy().tolist())
                if classification in votes:
                    votes[classification] += 1
                else:
                    votes[classification] = 1
        if self.voting_method == 'soft':
             for neighbor, distance in neighbours:
                classification = tuple(neighbor[self.classes_to_determine].to_numpy().tolist())
                weight = 1 / (distance + 1e-5) 
                votes[classification] = votes.get(classification, 0) + weight

        return max(votes, key=votes.get)
    

    def classify_test_set(self, control, test):
        classifications = []
        for _, test_point in test.iterrows():
            classification = self.classify_point(test_point, control)
            classifications.append(classification)
        return classifications


    def driver(self, normalize_data = True, calculate_nonnormalized = False):
        self.preprocess_data()
        shuffled_sets = self.shuffle(normalize=normalize_data)
        classified_tests = []
        for key, shuffled_set in shuffled_sets.items():
            if key == 'regular' and not calculate_nonnormalized: continue
            control, test = shuffled_set
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
            classified_tests.append(classified_test)
        return classified_tests

    def analyze_results(self, control, test, classifications, key = 'regular'):
        score = 0
        for actual, expected in zip(test[self.classes_to_determine].iterrows(), classifications):
            actual = tuple(actual[1].to_numpy().tolist())
            if actual == expected: score += 1
        # for actual, expected in zip(test[self.classes_to_determine]
        accuracy = score * 100 / len(test)
        normalized = f'normalized using {self.normalization_method}' if key == 'normalized' else '' 
        print(f'Method: {self.knn_method} (using {key} data, {normalized}), metric: {self.distance_metric}, accuracy: {accuracy}%')

        
    def remove_unknown_properties(self, dataset):
        return dataset[dataset.columns[self.keep_mask]]
