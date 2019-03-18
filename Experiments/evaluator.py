from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import cross_val_score
import numpy as np

class evaluator:
    classifier = None
    n_neighbors = 0
    
    def __init__(self,n_neighbors=1):        
        self.classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.n_neighbors = n_neighbors
    
    def fit(self,X,y):
        self.classifier.fit(X,y)
        
    def evaluate(self, X ,y_real):
        y_pred = []
        
        
        for i in X:
            y_pred.append(self.classifier.predict([i]))
        y_pred = np.array(y_pred)
        
        acc = metrics.accuracy_score(y_pred,y_real)
        pre = metrics.precision_score(y_pred,y_real)
        rec = metrics.recall_score(y_pred,y_real)
        f1 = metrics.f1_score(y_pred,y_real)
        
        return([acc,pre,rec,f1])

    def evaluate_with_kfold(self,X,y,n_folds=10):
        classif = KNeighborsClassifier(n_neighbors = self.n_neighbors)
        score_train = cross_val_score(classif ,X,y, cv=n_folds)
        
        return(np.mean(score_train),np.std(score_train))
        