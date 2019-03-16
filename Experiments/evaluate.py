import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import metrics
import numpy as np
from sklearn.neighbors import KNeighborsClassifier


from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels



class Evaluate():
    X_train, X_test, y_train, y_test = None, None, None, None
    classifier = None
    score_train = None
    y_pred = None
    
    
    def __init__(self,X,y,test_size=0.1):
        """ CONSTRUCTOR
        Input (X = Dataset, y = label, [test_size = 0.1])
        test_size is the size in percentage of the dataset that is used as test set.
        """ 
        ## split the dataset
        Evaluate.X_train, Evaluate.X_test, Evaluate.y_train, Evaluate.y_test = train_test_split(X, y, test_size=test_size, random_state=0)
        Evaluate.y_test = [[i] for i in Evaluate.y_test]
        Evaluate.y_test = np.ravel(Evaluate.y_test)
        Evaluate.y_train = [[i] for i in Evaluate.y_train]
        Evaluate.y_train = np.ravel(Evaluate.y_train)
        
    def fit(self, n_neighb=1, k_fold_validation=10):
        """ FIT
        It fit the model KNN on the test size
        Input ( [n_neighb=1], [k_fold_validation=10])
        n_neighb = number of neighbors used to train KKN
        k_fold_validation = number of fold for the validation 
        Return: model, sore_train = score obtained in each fold on k_fold_validation 
        """ 
        classifier = KNeighborsClassifier(n_neighbors=n_neighb)
        # cross validation to get scores
        Evaluate.score_train = cross_val_score(classifier , Evaluate.X_train, Evaluate.y_train,  cv=k_fold_validation)
        # train the whole Model
        classifier.fit(Evaluate.X_train,Evaluate.y_train)

        # save the Model
        Evaluate.classifier = classifier
        return( Evaluate.classifier, Evaluate.score_train)
    
    def predict(self):
        """ PREDICT
        Return: the prediction on the test set
        """ 
        y_pred = []
        for i in Evaluate.X_test:
            y_pred.append(Evaluate.classifier.predict([i]))
        
        Evaluate.y_pred = y_pred
        return(y_pred)
    
    def evaluate(self):    
        """ EVALUATE
        It return the accuracy of the model on the test set
        """ 
        accuracy = metrics.accuracy_score(Evaluate.y_test,Evaluate.y_pred)
        return(accuracy)

    def plot(self, h = .01):
        """ PLOT
        It plots the test points on the boundary coming from the fit
        """ 
        #h = .001
        # Plot the decision boundary. For that, we will assign a color to each

        x_border = (Evaluate.X_train[:, 0].mean()/len(Evaluate.X_train))*10
        y_border = (Evaluate.X_train[:, 1].mean()/len(Evaluate.X_train))*10

        x_min, x_max = Evaluate.X_train[:, 0].min() - x_border, Evaluate.X_train[:, 0].max() + x_border
        y_min, y_max = Evaluate.X_train[:, 1].min() - y_border, Evaluate.X_train[:, 1].max() + y_border
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        cmap_background = ListedColormap(['#0000FF','#F0FF00', '#FF0000', '#B4FFB1', '#FFB5FE', '#B5FDFF'])
        cmap_points = ListedColormap(['#0000FF','#F0FF00', '#F0FF00', '#B4FFB1', '#FFB5FE', '#B5FDFF'])

        # Obtain labels for each point in mesh. Use last trained Model.
        Z = Evaluate.classifier.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)


        plt.figure(figsize=(8, 8)) 
        plt.pcolormesh(xx, yy, Z, cmap=cmap_background)

        # Plot also the training points
        plt.scatter(Evaluate.X_test[:, 0], Evaluate.X_test[:, 1], c=Evaluate.y_test, cmap=cmap_points,edgecolor='k', s=20)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.title("Clusterd points in 2d using kNN")
        plt.show()

        
    def plot_confusion_matrix(self,cmap=plt.cm.Blues):
        """
        This function plots the normalized confusion matrix.
        """
        
        y_true = Evaluate.y_test
        y_pred = Evaluate.y_pred
        classes = np.unique(y_true)

        title = 'Normalized confusion matrix'
     

        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        # Only use the labels that appear in the data
       

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")

        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               # ... and label them with the respective list entries
               xticklabels=classes, yticklabels=classes,
               title=title,
               ylabel='True label',
               xlabel='Predicted label')

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        fmt = '.2f' 
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        
        plt.show()




    def report(self):
        """ REPORT
        produce a report containing information on K-fold validation,
        informations about test and a plot of the test point 
        """ 

        #### k fold cross validation
        print("Scores in bulding the model using K-fold validations:")
        for i in Evaluate.score_train:
            print("\t %.4f" %i)
        print("\n\t mean: %.4f" %np.mean(Evaluate.score_train))
        print("\t variance: %.4f" %np.var(Evaluate.score_train))

        plt.plot(Evaluate.score_train)
        plt.scatter(np.arange(0,10),Evaluate.score_train)
        plt.title("Score in K-fold validation")
        plt.xlabel("Group")
        plt.ylabel("Score")        
        plt.show()

        #### test set
        print("\n \n Results on test Set:")
        print("\t Accuracy: %.4f" % metrics.accuracy_score(Evaluate.y_test,Evaluate.y_pred))
        print("\t Precision: %.4f" % metrics.precision_score(Evaluate.y_test,Evaluate.y_pred,))
        print("\t Recall: %.4f" % metrics.recall_score(Evaluate.y_test,Evaluate.y_pred))
        print("\t F1_score: %.4f" % metrics.f1_score(Evaluate.y_test,Evaluate.y_pred))
        Evaluate.plot_confusion_matrix(self)
        
        #### plot data
        Evaluate.plot(self)
        