import pandas
import math
from queue import PriorityQueue

# locals
import MTree

class KNN:
 
    def __init__(self, path, k):
        self.df = pandas.read_csv(path)
        self.df = self.df.dropna()
        self.k = k

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
                keep = MTree.MTree()
                keep.insert(row, keep.root)
            else:
                neighbours = keep.KNN_search(row, self.k)
                classification = self.cast_votes(neighbours)
                if classification != row.iloc[-1]:
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

        print(f'Odrzuconych indeksów: {len(discard)}')

    def classify_point(self, point, control):
        neighbours = control.KNN_search(point, self.k)
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
            classification = neighbor.iloc[-1]
            if classification in votes:
                votes[classification] += 1
            else:
                votes[classification] = 1
        
        return max(votes.items(), key=lambda x: x[1])[0]

    def driver(self):
        self.preprocess_data()
        control, test = self.shuffle()
        self.CNN_reduction(control)

knn = KNN('student-mat.csv', 8)
knn.driver()