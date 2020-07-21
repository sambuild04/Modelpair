import numpy as np

class Preprocessing:
    def __init__(self, dataset):
        self.dataset = dataset
    def preprocess(self):
        import numpy as np
        import matplotlib.pyplot as plt
        import pandas as pd

        dataset = pd.read_csv(self.dataset)
        X = dataset.iloc[:, :-1].values
        y = dataset.iloc[:, -1].values


        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        print('Data Preprocessing Is Done!')

        return X_train, X_test, y_train, y_test
