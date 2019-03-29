import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
class Visualizator:
    '''Plot the results of an embedding
    
    Attributes:
        dimensions ([int]): an array of the tested dimensions
        n_classifiers (int): can be 1 or 2 and rappresent the number of calssifiers used.
    '''
 
    dim = []
    accuracy = []
    precision = [] 
    recall = []
    f1 = []
    accuracy_1 = []
    precision_1 = [] 
    recall_1 = []
    f1_1 = []
    pos = 0
    def __init__(self, dimensions, n_classifiers):
        
        self.dim = dimensions
        self.accuracy = [[] for i in dimensions]
        self.precision = [[] for i in dimensions]
        self.recall = [[] for i in dimensions]
        self.f1 = [[] for i in dimensions]
        if (n_classifiers == 2):
            self.accuracy_1 = [[] for i in dimensions]
            self.precision_1 = [[] for i in dimensions]
            self.recall_1 = [[] for i in dimensions]
            self.f1_1 = [[] for i in dimensions]

    
    def add_metrics(self,acc,pre,rec,f,acc_1=0,pre_1=0,rec_1=0,f_1=0):
        '''Add the computed metrics to the class.
        
         Args:
            acc ([int]): array of accuracy obtained with k-fold validations
            pre ([int]): array of precision obtained with k-fold validations
            rec ([int]): array of recall obtained with k-fold validations
            f1 ([int]): array of f1_score obtained with k-fold validations
            acc_1 ([int])[0]: array of accuracy obtained with k-fold validations with the second model
            pre_1 ([int])[0]: array of precision obtained with k-fold validations with the second model
            rec_1 ([int])[0]: array of recall obtained with k-fold validations with the second model
            f1_1 ([int])[0]: array of f1_score obtained with k-fold validations with the second model
        
        '''
        pos = self.pos
        self.accuracy[pos] = self.accuracy[pos] + acc
        self.precision[pos] = self.precision[pos] + pre
        self.recall[pos] = self.recall[pos] + rec
        self.f1[pos] = self.f1[pos] + f
        if(not acc_1 == 0):    # two classifiers
            self.accuracy_1[pos] = self.accuracy_1[pos] + acc_1
            self.precision_1[pos] = self.precision_1[pos] + pre_1
            self.recall_1[pos] = self.recall_1[pos] + rec_1
            self.f1_1[pos] = self.f1_1[pos] + f_1
        
        
        if (self.pos == len(self.dim)-1):
            self.pos = 0
        else:
            self.pos = self.pos + 1
        
    
    def summarize(self,size_train,size_test,method_name_1="",method_name_2=""):
        '''Print a summary of the classification.
        
        To get a summary of both methods, specify both methods name
        
        Args:
            size_train(int) : size of the train set
            size_test(int) : size of the test set
            method_name_1(String) : Title of the method 1
            method_name_2(String) : Title of the method 2
        
        '''
        
        print("Dimensions: ",self.dim)
        print("")
        print("Train set size: ",size_train)
        print("Test set size: ",size_test)
        acc_mean, acc_25, acc_75 = self.mean_std(self.accuracy)
        pre_mean, pre_25, pre_75 = self.mean_std(self.precision)    
        rec_mean, rec_25, rec_75 = self.mean_std(self.recall)
        f1_mean, f1_25, f1_75 = self.mean_std(self.f1)
        
        
        print("*****************************")
        print(method_name_1)
        acc_mean = list(acc_mean)
        acc_mean = ["Accuracy"] + acc_mean
        pre_mean = list(pre_mean)
        pre_mean = ["Precision"] + pre_mean
        rec_mean = list(rec_mean)
        rec_mean = ["Recall"] + rec_mean
        f1_mean = list(f1_mean)
        f1_mean = ["F1"] + f1_mean
        print (tabulate([acc_mean,pre_mean,rec_mean,f1_mean], headers=self.dim))
        
        if (not method_name_2 == ""):
            acc_mean, acc_25, acc_75 = self.mean_std(self.accuracy_1)
            pre_mean, pre_25, pre_75 = self.mean_std(self.precision_1)    
            rec_mean, rec_25, rec_75 = self.mean_std(self.recall_1)
            f1_mean, f1_25, f1_75 = self.mean_std(self.f1_1)


            print("*****************************")
            print(method_name_2)
            acc_mean = list(acc_mean)
            acc_mean = ["Accuracy"] + acc_mean
            pre_mean = list(pre_mean)
            pre_mean = ["Precision"] + pre_mean
            rec_mean = list(rec_mean)
            rec_mean = ["Recall"] + rec_mean
            f1_mean = list(f1_mean)
            f1_mean = ["F1"] + f1_mean
            print (tabulate([acc_mean,pre_mean,rec_mean,f1_mean], headers=self.dim))


    def mean_std(self, metric):
        '''Compute mean and percentile of an array of metrics.
        
        Args:
            metric ([float]): an Array of metrics computed in k fold validations
            
        Return:
            (mean_res, pre_25, per_75) mean of metrics, percentile 25 and percentile 75
        '''
        mean_res = []
        per_25 = []
        per_75 = []
        for i in metric:
            mean_res.append(np.mean(i))
            per_25.append(np.percentile(i,25))
            per_75.append(np.percentile(i,75))

        mean_res = np.array(mean_res)
        per_25 = np.array(per_25)
        per_75 = np.array(per_75)

        return (mean_res, per_25, per_75)
    
    
    
    def balance_targhet_test(self,y_test):
        '''Count the number of positive and negative targhets.
        
        Args:
            y_test([int]): list of test targhet
            
        Return:
            (y_balance): A pair (n_positive_semples, n_negative_samples)
        '''
        
        y_balance = [0,0]
        for i in y_test:
            if (i == 0):
                y_balance[0] = y_balance[0] + 1
            else:
                y_balance[1] = y_balance[1] + 1
                
        return(y_balance)
    
    
    def mixed_plot(self,y_test):
        '''Plot both classifiers results.
        
        Plot Accuracy, Precision, Recall, F1_score when the dimension increase.
        each metric is surrounded by its percentile
        
        Args:
            y_test([int]): list of test targhet
            
        '''
        
        a = self.accuracy
        p = self.precision
        r = self.recall
        f = self.f1

        a_1 = self.accuracy_1
        p_1 = self.precision_1
        r_1 = self.recall_1
        f_1 = self.f1_1            
        acc_mean, acc_25, acc_75 = self.mean_std(a)
        acc_mean_1, acc_25_1, acc_75_1 = self.mean_std(a_1)
        pre_mean, pre_25, pre_75 = self.mean_std(p)
        pre_mean_1, pre_25_1, pre_75_1 = self.mean_std(p_1)
        rec_mean, rec_25, rec_75 = self.mean_std(r)
        rec_mean_1, rec_25_1, rec_75_1 = self.mean_std(r_1)
        f1_mean, f1_25, f1_75 = self.mean_std(f)
        f1_mean_1, f1_25_1, f1_75_1 = self.mean_std(f_1)

        y_balance = self.balance_targhet_test(y_test)

        
        plt.figure(1, figsize=(10,10))

        plt.subplot(321)
        plt.title("Accuracy")
        # the 1 sigma upper and lower analytic population bounds
        lower_bound = acc_25
        upper_bound = acc_75
        lower_bound_1 = acc_25_1
        upper_bound_1 = acc_75_1

        plt.semilogx(self.dim, acc_mean, lw=2, label='Accuracy KNN', color='blue')
        plt.scatter(self.dim, acc_mean, color='blue')
        plt.semilogx(self.dim, acc_mean_1, lw=2, label='Accuracy Rand For.', color='red')
        plt.scatter(self.dim, acc_mean_1, color='red')

        base_line = max(y_balance)/np.sum(y_balance)

        plt.hlines(base_line, 0, self.dim[-1], color='red',linestyles= 'dashed')
        plt.fill_between(self.dim, lower_bound, upper_bound, facecolor='blue', alpha=0.3,
                        label='precentile 25-75 KNN')
        
        plt.fill_between(self.dim, lower_bound_1, upper_bound_1, facecolor='#FF8484', alpha=0.3,
                        label='precentile 25-75 Rand For.')
        plt.legend(loc='lower right')
        plt.ylim(0,1)
        plt.grid()


        

        plt.subplot(322)
        plt.title("Precision")
        # the 1 sigma upper and lower analytic population bounds
        lower_bound = pre_25
        upper_bound = pre_75
        lower_bound_1 = rec_25_1
        upper_bound_1 = rec_75_1

        plt.semilogx(self.dim, pre_mean, lw=2, label='precision KNN', color='blue')
        plt.scatter(self.dim, pre_mean, color='blue')
        plt.semilogx(self.dim, pre_mean_1, lw=2, label='precision Rand For.', color='red')
        plt.scatter(self.dim, pre_mean_1, color='blue')
        
        plt.fill_between(self.dim, lower_bound, upper_bound, facecolor='#9B8BFF', alpha=0.3,
                        label='precentile 25-75 KNN')
        plt.fill_between(self.dim, lower_bound_1, upper_bound_1, facecolor='#FF8484', alpha=0.3,
                        label='precentile 25-75 Rand For.')
        plt.legend(loc='lower right')
        plt.ylim(0,1)
        plt.grid()


        plt.subplot(323)
        plt.title("Recall")
        lower_bound = rec_25
        upper_bound = rec_75
        lower_bound_1 = rec_25_1
        upper_bound_1 = rec_75_1

        plt.semilogx(self.dim, rec_mean, lw=2, label='Recall KNN', color='blue')
        plt.scatter(self.dim, rec_mean, color='blue')
        plt.semilogx(self.dim, rec_mean_1, lw=2, label='Recall Rand For.', color='red')
        plt.scatter(self.dim, rec_mean_1, color='red')
        plt.fill_between(self.dim, lower_bound, upper_bound, facecolor='#9B8BFF', alpha=0.3,
                        label='precentile 25-75 KNN')
        plt.fill_between(self.dim, lower_bound_1, upper_bound_1, facecolor='#FF8484', alpha=0.3,
                        label='precentile 25-75 Rand For.')
        plt.legend(loc='lower right')
        plt.ylim(0,1)
        plt.grid()

        
        plt.subplot(324)
        plt.title("F1")
        lower_bound = f1_25    
        upper_bound = f1_75
        lower_bound_1 = f1_25_1    
        upper_bound_1 = f1_75_1

        plt.semilogx(self.dim, f1_mean, lw=2, label='F1 KNN', color='blue')
        plt.scatter(self.dim, f1_mean, color='blue')
        plt.semilogx(self.dim, f1_mean_1, lw=2, label='F1 Rand. For.', color='red')
        plt.scatter(self.dim, f1_mean_1, color='red')
        plt.fill_between(self.dim, lower_bound, upper_bound, facecolor='#9B8BFF', alpha=0.3,
                        label='precentile 25-75 KNN')
        plt.fill_between(self.dim, lower_bound_1, upper_bound_1, facecolor='#FF8484', alpha=0.3,
                        label='precentile 25-75 Rand For.')
        plt.legend(loc='lower right')
        plt.ylim(0,1)
        plt.grid()
        plt.show()

        

    def single_plot(self,y_test,model=1):
        '''Plot classifier results.
        
        Plot Accuracy, Precision, Recall, F1_score when the dimension increase.
        each metric is surrounded by its percentile
        
        Args:
            y_test([int]): list of test targhet
            model (int)[1]: 1 or 2 
            
        '''
        
        print("")
        print("**********************************************************")
        print("Model: ",model)
        print("")
        if (model == 1):
            a = self.accuracy
            p = self.precision
            r = self.recall
            f = self.f1
        elif (model == 2):
            a = self.accuracy_1
            p = self.precision_1
            r = self.recall_1
            f = self.f1_1            
        acc_mean, acc_25, acc_75 = self.mean_std(a)
        pre_mean, pre_25, pre_75 = self.mean_std(p)
        rec_mean, rec_25, rec_75 = self.mean_std(r)
        f1_mean, f1_25, f1_75 = self.mean_std(f)

        y_balance = self.balance_targhet_test(y_test)

        
        plt.figure(1, figsize=(10,10))

        plt.subplot(321)
        plt.title("Accuracy")
        # the 1 sigma upper and lower analytic population bounds
        lower_bound = acc_25
        upper_bound = acc_75

        plt.semilogx(self.dim, acc_mean, lw=2, label='Accuracy', color='blue')
        plt.scatter(self.dim, acc_mean, color='blue')

        base_line = max(y_balance)/np.sum(y_balance)

        plt.hlines(base_line, 0, self.dim[-1], color='red',linestyles= 'dashed')
        plt.fill_between(self.dim, lower_bound, upper_bound, facecolor='#9B8BFF', alpha=0.5,
                        label='precentile 25-75')
        plt.legend(loc='lower right')
        plt.ylim(0,1)
        plt.grid()



        plt.subplot(322)
        plt.title("Precision")
        # the 1 sigma upper and lower analytic population bounds
        lower_bound = pre_25
        upper_bound = pre_75

        plt.semilogx(self.dim, pre_mean, lw=2, label='Precision', color='blue')
        plt.scatter(self.dim, pre_mean, color='blue')
        plt.fill_between(self.dim, lower_bound, upper_bound, facecolor='#9B8BFF', alpha=0.5,
                        label='std')
        plt.legend(loc='lower right')
        plt.ylim(0,1)
        plt.grid()


        plt.subplot(323)
        plt.title("Recall")
        lower_bound = rec_25
        upper_bound = rec_75

        plt.semilogx(self.dim, rec_mean, lw=2, label='Recall', color='blue')
        plt.scatter(self.dim, rec_mean, color='blue')
        plt.fill_between(self.dim, lower_bound, upper_bound, facecolor='#9B8BFF', alpha=0.5,
                        label='std')
        plt.legend(loc='lower right')
        plt.ylim(0,1)
        plt.grid()

        plt.subplot(324)
        plt.title("F1")
        lower_bound = f1_25    
        upper_bound = f1_75

        plt.semilogx(self.dim, f1_mean, lw=2, label='F1', color='blue')
        plt.scatter(self.dim, f1_mean, color='blue')
        plt.fill_between(self.dim, lower_bound, upper_bound, facecolor='#9B8BFF', alpha=0.5,
                        label='std')
        plt.legend(loc='lower right')
        plt.ylim(0,1)
        plt.grid()
        plt.show()
