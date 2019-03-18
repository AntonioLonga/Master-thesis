from eden.graph import vectorize
import numpy as np



class embedder:
    model_high_to_medium = None
    model_medium_to_small = None
    """ Constructor that take in input two models"""
    def __init__(self, model_medium_to_small, model_high_to_medium="vectorize"):
        """ Constructor that take in input two models:
            model_high_to_medium -> first reduction from really high space to a medium one"""
        # instanziate first model
        # Default = vectorize
        embedder.model_medium_to_small = model_medium_to_small
        # instanziate second model
        embedder.model_high_to_medium = model_high_to_medium
        
    def fit(self, graphs):
        # fit the first model
        X = self.__fit_high_to_medium(graphs)
        # fit the second model
        self.__fit_medium_to_small(X)
        
    def transform(self, graphs):
        # use the first model to predict data
        Xd = self.__transform_high_to_medium(graphs)
        # use the second modle to predict data coming from first model
        y_pred = self.__transform_medium_to_small(Xd)
        return(y_pred)
    
    
    def __fit_high_to_medium(self, graphs):
        if (embedder.model_high_to_medium == "vectorize"):
            X = vectorize(graphs, complexity=12)
        else:
            ###### something like
            # X = model_high_to_medium.fit(graphs)
            X = None
        return (X)
    
    def __transform_high_to_medium(self, graphs):
        if (embedder.model_high_to_medium == "vectorize"):
            X = vectorize(graphs, complexity=12, )
        else:
            ###### something like
            # X = model_high_to_medium.transform(graphs)
            X = None
        return (X)
    
    def __fit_medium_to_small(self, X):
        embedder.model_medium_to_small.fit(X)
        
    def __transform_medium_to_small(self, X):
        return(embedder.model_medium_to_small.transform(X))