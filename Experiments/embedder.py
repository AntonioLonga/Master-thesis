class Embedder:
    
    estimatros = []
    
    def __init__(self, estim):
        self.estimatros = estim
        
    def fit(self,X,y):
        X_res = X
        for e in range(0,len(self.estimatros)):
            if (e == len(self.estimatros)-1):
                self.estimatros[e].fit(X_res,y)
                return(self)
            else:
                X_res = self.estimatros[e].fit(X_res,y).transform(X_res)

    def transform(self,X):
        X_res = X
        
        for e in self.estimatros:
            X_res = e.transform(X_res)
            
        return(X_res)
        

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
        