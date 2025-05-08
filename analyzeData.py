import pandas
import matplotlib.pyplot as plt
import seaborn as sns

class analyzeData:
    def __init__(self,path):
        self.df = pandas.read_csv(path)
        self.df = self.df.dropna()

    
    def plot(self):
        """
        Plot the data using seaborn
        """
        sns.pairplot(self.df)
        plt.show()

    def describe(self):
        """
        Print the description of the data
        """
        print(self.df.describe())

    def correlation(self):
        """
        Print the correlation of the data
        """
        print(self.df.corr())
        sns.heatmap(self.df.corr(), annot=True)
        plt.show()

    def preprocess_data(self):
        """
        Define dictionaries containing all non-number features so that they can be mapped to numbers.
        """
        for column in self.df:
            self.df = self.df.apply(lambda x: pandas.factorize(x)[0])
           
        

dataAnalytics = analyzeData('student-mat.csv')
dataAnalytics.preprocess_data()
#dataAnalytics.plot()
dataAnalytics.describe()
dataAnalytics.correlation()


