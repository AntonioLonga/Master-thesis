from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import cross_val_score
import numpy as np

class Evaluator:
    '''
    This class use K neighbours Classifier to valuate an embedding. 
    It has a construcctor, that thakes in input the number of neighbour, and
    a method that compute accuracy percision recall and f1 score using 
    n_folds cross validation.
    '''
    classifier = None
    n_neighbors = 0
    
    def __init__(self,n_neighbors=1):        
        self.classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.n_neighbors = n_neighbors


    def performance_with_kfold(self,X,y,n_folds=10):
        '''
        It takes in input a set X and it's label and the number of folds (default = 10)
        It return an array of lists [accuracy, precision, recall, f1] 
        length n_folds.
        '''

        classif = KNeighborsClassifier(n_neighbors = self.n_neighbors)
        
        accuracy_train = list(cross_val_score(classif ,X,y, cv=n_folds, 
                                         scoring = metrics.make_scorer(metrics.accuracy_score)))
        precision_train = list(cross_val_score(classif ,X,y, cv=n_folds, 
                                          scoring = metrics.make_scorer(metrics.precision_score)))
        recall_train = list(cross_val_score(classif ,X,y, cv=n_folds, 
                                       scoring = metrics.make_scorer(metrics.recall_score)))
        f1_train = list(cross_val_score(classif ,X,y, cv=n_folds, 
                                   scoring = metrics.make_scorer(metrics.f1_score)))
        
        return([accuracy_train,precision_train,recall_train,f1_train])
        