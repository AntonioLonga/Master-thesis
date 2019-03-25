class Embedder:
    estimator_graph_to_high = None
    estimator_high_to_medium = None
    estimator_medium_to_small = None
    n_estimator = 2
    
    def __init__(self, graph_to_high, high_to_medium, medium_to_small=None):
        self.estimator_graph_to_high = Model(graph_to_high)
        self.estimator_high_to_medium = Model(high_to_medium)
        self.estimator_medium_to_small = Model(medium_to_small)
        if (medium_to_small != None):
            self.n_estimator = 3
        
    def fit(self,X,y):
        
        if (self.n_estimator == 3):
            X_high = self.estimator_graph_to_high.fit(X,y).transform(X)
            X_medium = self.estimator_high_to_medium.fit(X_high,y).transform(X_high)
            X_small = self.estimator_medium_to_small.fit(X_medium,y)
        else:
            X_high = self.estimator_graph_to_high.fit(X,y).transform(X)
            X_medium = self.estimator_high_to_medium.fit(X_high,y)
        
        return(self)
    
    def transform(self,X):
        
        if (self.n_estimator == 3):
            X_high = self.estimator_graph_to_high.transform(X)
            X_medium = self.estimator_high_to_medium.transform(X_high)
            X_small = self.estimator_medium_to_small.transform(X_medium)
        else:
            X_high = self.estimator_graph_to_high.transform(X)
            X_small = self.estimator_high_to_medium.transform(X_high)
        
        return(X_small)
        

class Model:
    estimator = None
    has_fit = True
    
    def __init__(self, estimator, has_fit=True):
        self.has_fit = has_fit
        self.estimator = estimator
    
    def fit(self,X,y):
        if (self.has_fit == True):
            self.estimator.fit(X,y)
        return(self)
    
    def transform(self,X):

        y_pred = self.estimator.transform(X)
        return(y_pred)
        