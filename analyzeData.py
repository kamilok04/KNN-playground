import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class analyzeData:
    def __init__(self,path,numerical_columns, category_columns):
        self.df = pd.read_csv(path)
        self.num_cols = numerical_columns
        self.cat_cols = category_columns
        self.df = self.df.dropna()

    def describe(self):
        """
        Print the description of the data
        """
        print(self.df.info())
        print(self.df.describe())
        print(self.df.head())

    def correlation(self):
        """
        Print the correlation matrix of the data
        """
        corr = self.df.corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm')
        plt.title('Correlation matrix')
        plt.show()

    def preprocess_data(self):
        """
        Define dictionaries containing all non-number features so that they can be mapped to numbers.
        """
        for column in self.df:
            self.df = self.df.apply(lambda x: pd.factorize(x)[0])

    def numericalDistribution(self):
        """
        Print the distribution of the numerical columns
        """
        for col in self.num_cols:
            sns.histplot(self.df[col], kde=True)
            plt.title(f'Distribution: {col}')
            plt.show()

    def uniqueValues(self):
        """
        Print the unique values of the categorical columns
        """
        for col in self.cat_cols:
            print(f'{col}: {self.df[col].unique()}')

    def categoricalDistribution(self):
        """
        Print the distribution of the categorical columns compared to heart disease
        """
        for col in self.cat_cols:
            sns.countplot(data=self.df, x=col, hue='HeartDisease')
            plt.title(f'{col} distribution compared to HeartDisease')
            plt.show()

    def pairPlot(self, columns):
        """
        Print the pair plot of chosen data
        """
        sns.pairplot(self.df, hue='HeartDisease', vars=columns)
        plt.show()

    def boxPlot(self, x_column, y_column):
        """
        Print the box plot of the chosen data
        """
        sns.boxplot(x=x_column, y=y_column, data=self.df)
        plt.title(f'{x_column} vs {y_column}')
        plt.show()

    def driver(self):
        self.describe()
        self.uniqueValues()
        self.categoricalDistribution()
        self.numericalDistribution()
        self.pairPlot(['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak'])
        self.boxPlot('HeartDisease', 'Cholesterol')
        self.preprocess_data()
        self.correlation()

dataAnalytics = analyzeData('heart.csv',
['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak','HeartDisease'],
['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope','FastingBS'])


dataAnalytics.driver()
# dataAnalytics.preprocess_data()
# dataAnalytics.correlation()



