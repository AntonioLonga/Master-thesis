
from sklearn import metrics
from sklearn.model_selection import cross_val_score
import numpy as np

class Evaluator:
    '''
    This class evaluate an embedding
    
    Attributes:
        classifiers ([]):  a list of clasisfier
        
    Example:
    
        mod1 = RandomForestClassifier(n_estimators=1)
        mod2 = KNeighborsClassifier(n_neighbors=1)
        
        Evaluatot([mod1,mod2])
    '''
    
    classifier = None
    
    def __init__(self, classif):        
        self.classifier = classif


    def performance_with_kfold(self,X,y,n_folds=10):
        '''
        Compute some metrics on a dataset in k-fold corss validations.
        
        Args: 
            X ([[int,..int]..[int,..int]]) : Point to classify
            y ([int]) : real targhet
            n_folds (int)[10] : number of folds
            
        Return:
            [accuracy,precision,recall,f1] : array of array
        '''

        classif = self.classifier

        
        accuracy_train = list(cross_val_score(classif ,X,y, cv=n_folds, 
                                         scoring = metrics.make_scorer(metrics.accuracy_score)))
        precision_train = list(cross_val_score(classif ,X,y, cv=n_folds, 
                                          scoring = metrics.make_scorer(metrics.precision_score)))
        recall_train = list(cross_val_score(classif ,X,y, cv=n_folds, 
                                       scoring = metrics.make_scorer(metrics.recall_score)))
        f1_train = list(cross_val_score(classif ,X,y, cv=n_folds, 
                                   scoring = metrics.make_scorer(metrics.f1_score)))
        
        return([accuracy_train,precision_train,recall_train,f1_train])
        