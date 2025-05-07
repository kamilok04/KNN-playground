import pandas
import math

class KNN:
    def __init__(self, path):
        self.df = pandas.read_csv(path)
        self.df = self.df.dropna()
        print(self.df.describe())
        self.preprocess_data()

    def preprocess_data(self):
        """
        Define dictionaries containing all non-number features so that they can be mapped to numbers.
        """
        for column in self.df:
            self.df = self.df.apply(lambda x: pandas.factorize(x)[0])
        print(self.df.describe())
    
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
        """

        keep = []
        discard = []
        # Implement the CNN reduction algorithm
        for index, row in control.iterrows():
            if not keep:
                keep.append(row)
           
                
        return pandas.DataFrame(keep)

    def classify(self, control):
        control = self.CNN_reduction(control)
        control.describe()

    def classify_point(self, point, control):
        """Assign a class to a single test point; necessary for CNN reduction

        Args:
            point (_type_): _description_
            control (_type_): _description_
        """


    def driver(self):
        control, test = self.shuffle()
        self.classify(control)

knn = KNN('student-mat.csv')
knn.driver()