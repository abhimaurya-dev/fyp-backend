import numpy as np
import pandas as pd
import pickle
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer


def train_save_model():
  dataset = pd.read_csv("./dataset/Crop_recommendation.csv")
  X = dataset.iloc[:,:-1]
  Y = dataset.iloc[:,-1].values
  imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
  X = imputer.fit_transform(X)
  X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=42)
  classifier = GaussianNB()
  classifier.fit(X_train,Y_train)
  with open('./trainedModel/trained_model','wb') as f:
    pickle.dump(classifier,f)

train_save_model()


