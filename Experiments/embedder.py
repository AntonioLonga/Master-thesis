class Embedder:
    '''Embed a set of graphs in a real space.
    
    Embedder apply sequencially the models.
    Example:
        estim = [model1,model2,model3]
        
        x1 = model1.fit(graphs,targhet).transform(x1)
        x2 = model2.fit(x1, targhet).transform(x2)
        x3 = model3.fit(x2, targhet)
        
    Attributes:
        estim ([Model]): a list of models 
        
    '''
    estimatros = []
    
    def __init__(self, estim):
        self.estimatros = estim
        
    def fit(self,X,y):
        '''fit the data on the models.
        
              
        Args:
            X ([nx.graph]) : array of graphs
            y ([int]) : targhet
            
        Returns:
            (self)
        '''
        X_res = X
        for e in range(0,len(self.estimatros)):
            if (e == len(self.estimatros)-1):
                X_res = self.estimatros[e].fit(X_res,y).transform(X_res)
                return(self)
            else:
                X_res = self.estimatros[e].fit(X_res,y).transform(X_res)

                

    def transform(self,X):
        '''Transform set of graphs in points in a space.
              
        Args:
            X ([nx.graph]) : array of graphs
            
        Returns:
            (X_res) : predicted targhet
        '''
        X_res = X
        
        for e in self.estimatros:
            X_res = e.transform(X_res)
            
        return(X_res)
        

class Model:
    '''Contains an Estimator

    Attributes:
        estimator (Obj): an estimator
        has_fit (boolena [True]): specify if a model has the fit fucntion.
                                For example, Vectorize hasn't fit. 
        
    '''
    estimator = None
    has_fit = True
    
    def __init__(self, estimator, has_fit=True):
        self.has_fit = has_fit
        self.estimator = estimator
    
    def fit(self,X,y):
        '''Fit the model usign X and y
              
        Args:
            X ([nx.graph]) : array of graphs
            y ([int]) : array of targhets
            
        Returns:
            (self)
        '''
        if (self.has_fit == True):
            self.estimator.fit(X,y)
        return(self)
    
    def transform(self,X):
        '''Transform X in points in a space.
              
        Args:
            X ([nx.graph] or [int]) : array of graphs
            
        Returns:
            (X_pred) : predicted targhet
        '''

        y_pred = self.estimator.transform(X)
        return(y_pred)
        