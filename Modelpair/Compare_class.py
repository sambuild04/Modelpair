from .Preprocess import Preprocessing
from .SK_decision import SK_decision
from .SK_knn import SK_knn
import sklearn

class Compare_class:
    def __init__(self, dataset):
        self.dataset = dataset

    def generate_compare(self):
        decision_score = SK_decision(self.dataset).generate_skdt()
        knn_score= SK_knn(self.dataset).generate_sknn()
        print('Decision Tree score is ', decision_score, 'and KNN score is', knn_score)
