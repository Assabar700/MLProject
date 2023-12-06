import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

class MyDataProcessor:
    def __init__(self, data_path, labelsCol="species"):
        self.data_path = data_path
        self.labelsCol = labelsCol
        self.df = pd.read_csv(data_path)
        self.df[labelsCol] = LabelEncoder().fit(self.df[labelsCol]).transform(self.df[labelsCol])
        self.trainSize = 0.7
        self.XTrain, self.YTrain, self.XTest, self.YTest = self.split_data()

    def split_data(self):
        dfTrain, dfTest = train_test_split(self.df, train_size=self.trainSize, stratify=self.df[self.labelsCol], shuffle=True)

        YTrain = dfTrain[self.labelsCol].to_numpy()
        YTest = dfTest[self.labelsCol].to_numpy()

        XTrain = dfTrain.drop([self.labelsCol], axis=1).to_numpy()
        XTest = dfTest.drop([self.labelsCol], axis=1).to_numpy()

        return XTrain, YTrain, XTest, YTest
    

