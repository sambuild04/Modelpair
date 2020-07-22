from .Preprocess import Preprocessing
import sklearn


class SK_decision:
    def __init__(self, dataset):
        self.dataset = dataset
#when use, called(SK_decision('project_path'))
    def generate_skdt(self):
        X_train, X_test, y_train, y_test = Preprocessing(self.dataset).preprocess()
        from sklearn.tree import DecisionTreeClassifier
        classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
        classifier.fit(X_train, y_train)
        from sklearn.metrics import confusion_matrix, accuracy_score
        y_pred = classifier.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        accuracy_score(y_test, y_pred)
        print('The accuracy_score for sklearn decision tree is ', accuracy_score(y_test, y_pred))
        return accuracy_score(y_test, y_pred)
