
import numpy as np
from tabulate import tabulate
from scipy.stats import rankdata
import matplotlib.pyplot as plt
class Visualizator:
    '''Plot the results of an embedding
    
    Attributes:
        dimensions ([int]): an array of the tested dimensions
        n_classifiers (int): can be 1 or 2 and rappresent the number of calssifiers used.
    '''
 
    dim = []
    n_classifiers = 0
    accuracy = []
    precision = [] 
    recall = []
    f1 = []
    
    y_test=None
    
    models_names = []
     
    pos = 0
    def __init__(self, dimensions, n_classifiers, models_names):
        self.accuracy = []
        self.precision = [] 
        self.recall = []
        self.f1 = []
        
        
        self.models_names = models_names
        self.n_classifiers = n_classifiers
        self.dim = dimensions
        
        for i in range(0,n_classifiers):
            tmp = []
            for j in range(0,len(dimensions)):
                tmp.append([])
            self.accuracy.append(tmp.copy())
            self.precision.append(tmp.copy())
            self.recall.append(tmp.copy())
            self.f1.append(tmp.copy())
      
    
    def add_metrics(self,acc,pre,rec,f, classifier_number, dimension):

        self.accuracy[classifier_number][dimension] = self.accuracy[classifier_number][dimension] + acc
        self.precision[classifier_number][dimension] = self.precision[classifier_number][dimension] + pre
        self.recall[classifier_number][dimension] = self.recall[classifier_number][dimension] + rec
        self.f1[classifier_number][dimension] = self.f1[classifier_number][dimension] + f
    
    def rank(self,metric="accuracy",return_matrix=False):
        
        dim = self.dim
        res = []
        res = [[] for x in dim]
        if metric == "accuracy":
            for j in range(0,len(res)):
                for i in range(0,len(self.accuracy)):
                    res[j].append(1-np.mean(self.accuracy[i][j]))
        if metric == "precision":
            for j in range(0,len(res)):
                for i in range(0,len(self.precision)):
                    res[j].append(1-np.mean(self.precision[i][j]))
        if metric == "recall":
            for j in range(0,len(res)):
                for i in range(0,len(self.recall)):
                    res[j].append(1-np.mean(self.recall[i][j]))
        if metric == "f1":
            for j in range(0,len(res)):
                for i in range(0,len(self.f1)):
                    res[j].append(1-np.mean(self.f1[i][j]))

                
        for i in range(0,len(res)):
            res[i] = rankdata(res[i])


        models_name = self.models_names
        rank_model = [[i] for i in models_name]

        res_transpose = list(map(list, zip(*res)))
        for i in range(0,len(rank_model)):
            for j in res_transpose[i]:
                rank_model[i].append(j)
            rank_model[i].append(np.mean(res_transpose[i]))
            

        

        if return_matrix == True:
            return rank_model
        else:
            dims = dim.copy()
            dims.append("mean")
            print (tabulate(rank_model, headers=dims))

        
    def mean_percentile(self, metric):
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

        mean_res = np.around(mean_res, decimals=3) 
        per_25 = np.around(per_25, decimals=3) 
        per_75 = np.around(per_75, decimals=3) 

        return (mean_res, per_25, per_75)



    def mean_std(self, metric, std):
        '''Compute mean and percentile of an array of metrics.
        
        Args:
            metric ([float]): an Array of metrics computed in k fold validations
            
        Return:
            (mean_res, pre_25, per_75) mean of metrics, percentile 25 and percentile 75
        '''
        mean_res = []
        for i in metric:
            mean_res.append(np.mean(i))
            if (std == True):
                mean_res.append(np.std(i))

        mean_res = np.around(mean_res, decimals=3) 

        return (mean_res)
        
        
    
    def summary_old(self):
                
        print("Dimensions: ",self.dim)
        print("")
        for i in range(0,self.n_classifiers):
            acc_mean, acc_25, acc_75 = self.mean_percentile(self.accuracy[i])
            pre_mean, pre_25, pre_75 = self.mean_percentile(self.precision[i])    
            rec_mean, rec_25, rec_75 = self.mean_percentile(self.recall[i])
            f1_mean, f1_25, f1_75 = self.mean_percentile(self.f1[i])

        
            print("*****************************")
            print("Method: ",self.models_names[i])
            acc_mean = list(acc_mean)
            acc_mean = ["Accuracy"] + acc_mean
            pre_mean = list(pre_mean)
            pre_mean = ["Precision"] + pre_mean
            rec_mean = list(rec_mean)
            rec_mean = ["Recall"] + rec_mean
            f1_mean = list(f1_mean)
            f1_mean = ["F1"] + f1_mean
            print (tabulate([acc_mean,pre_mean,rec_mean,f1_mean], headers=self.dim))
      

    def summary(self,metric='accuracy',std=True,return_matrix=False):
        if (metric == "accuracy"):
            metric = self.accuracy
        elif(metric == "precision"):
            metric = self.precision
        elif(metric == "recall"):
            metric = self.recall
        else:
            metric = self.f1

        means= []
        for i in range(0,self.n_classifiers):
            
            acc_mean = self.mean_std(metric[i],std=std)
            acc_mean = list(acc_mean)
            acc_mean = acc_mean + [np.around(np.mean(acc_mean), decimals=3)]
            acc_mean = [self.models_names[i]] + acc_mean
            
            means.append(acc_mean)

        dims = []
        for i in self.dim:
            dims.append(i)
            if (std == True):
                dims.append("STD")
    
        if return_matrix == False:
            print (tabulate(means, headers=list(dims)+['mean']))
        else:
            return(means)
    
    
    
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
    
     

    def plot(self,model=0):
        if (type(model) == int):
            self.single_plot(model)
        else:
            self.mixed_plot(model[0],model[1])

    def single_plot(self,model):
        
        
        print("")
        print("**********************************************************")
        print("Dims: ",self.dim)
        print("Model: ",self.models_names[model])
        print("")
        a = self.accuracy
        p = self.precision
        r = self.recall
        f = self.f1

        acc_mean, acc_25, acc_75 = self.mean_percentile(self.accuracy[model])
        pre_mean, pre_25, pre_75 = self.mean_percentile(self.precision[model])    
        rec_mean, rec_25, rec_75 = self.mean_percentile(self.recall[model])
        f1_mean, f1_25, f1_75 = self.mean_percentile(self.f1[model])

        y_balance = self.balance_targhet_test(self.y_test)

        
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

    
    def mixed_plot(self,model_1,model_2):
        '''Plot both classifiers results.
        
        Plot Accuracy, Precision, Recall, F1_score when the dimension increase.
        each metric is surrounded by its percentile
        
        Args:
            y_test([int]): list of test targhet
            
        '''
        print("Dims: ",self.dim)
        print("Model: ",self.models_names[model_1])
        print("Model: ",self.models_names[model_2])
        
        a = self.accuracy
        p = self.precision
        r = self.recall
        f = self.f1
        
        acc_mean, acc_25, acc_75 = self.mean_percentile(self.accuracy[model_1])
        pre_mean, pre_25, pre_75 = self.mean_percentile(self.precision[model_1])    
        rec_mean, rec_25, rec_75 = self.mean_percentile(self.recall[model_1])
        f1_mean, f1_25, f1_75 = self.mean_percentile(self.f1[model_1])
        

        acc_mean_1, acc_25_1, acc_75_1 = self.mean_percentile(self.accuracy[model_2])
        pre_mean_1, pre_25_1, pre_75_1 = self.mean_percentile(self.precision[model_2])    
        rec_mean_1, rec_25_1, rec_75_1 =self.mean_percentile(self.recall[model_2])
        f1_mean_1, f1_25_1, f1_75_1 = self.mean_percentile(self.f1[model_2])

        y_balance = self.balance_targhet_test(self.y_test)

        
        plt.figure(1, figsize=(10,10))

        plt.subplot(321)
        plt.title("Accuracy")
        # the 1 sigma upper and lower analytic population bounds
        lower_bound = acc_25
        upper_bound = acc_75
        lower_bound_1 = acc_25_1
        upper_bound_1 = acc_75_1

        plt.semilogx(self.dim, acc_mean, lw=2, label=self.models_names[model_1], color='blue')
        plt.scatter(self.dim, acc_mean, color='blue')
        plt.semilogx(self.dim, acc_mean_1, lw=2, label=self.models_names[model_2], color='red')
        plt.scatter(self.dim, acc_mean_1, color='red')

        base_line = max(y_balance)/np.sum(y_balance)

        plt.hlines(base_line, 0, self.dim[-1], color='red',linestyles= 'dashed')
        plt.fill_between(self.dim, lower_bound, upper_bound, facecolor='blue', alpha=0.3)
        
        plt.fill_between(self.dim, lower_bound_1, upper_bound_1, facecolor='#FF8484', alpha=0.3)
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

        plt.semilogx(self.dim, pre_mean, lw=2,  label=self.models_names[model_1], color='blue')
        plt.scatter(self.dim, pre_mean, color='blue')
        plt.semilogx(self.dim, pre_mean_1, lw=2,  label=self.models_names[model_2], color='red')
        plt.scatter(self.dim, pre_mean_1, color='red')
        
        plt.fill_between(self.dim, lower_bound, upper_bound, facecolor='#9B8BFF', alpha=0.3)
        plt.fill_between(self.dim, lower_bound_1, upper_bound_1, facecolor='#FF8484', alpha=0.3)
        plt.legend(loc='lower right')
        plt.ylim(0,1)
        plt.grid()


        plt.subplot(323)
        plt.title("Recall")
        lower_bound = rec_25
        upper_bound = rec_75
        lower_bound_1 = rec_25_1
        upper_bound_1 = rec_75_1

        plt.semilogx(self.dim, rec_mean, lw=2,  label=self.models_names[model_1], color='blue')
        plt.scatter(self.dim, rec_mean, color='blue')
        plt.semilogx(self.dim, rec_mean_1, lw=2,  label=self.models_names[model_2], color='red')
        plt.scatter(self.dim, rec_mean_1, color='red')
        plt.fill_between(self.dim, lower_bound, upper_bound, facecolor='#9B8BFF', alpha=0.3)
        plt.fill_between(self.dim, lower_bound_1, upper_bound_1, facecolor='#FF8484', alpha=0.3)
        plt.legend(loc='lower right')
        plt.ylim(0,1)
        plt.grid()

        
        plt.subplot(324)
        plt.title("F1")
        lower_bound = f1_25    
        upper_bound = f1_75
        lower_bound_1 = f1_25_1    
        upper_bound_1 = f1_75_1

        plt.semilogx(self.dim, f1_mean, lw=2,  label=self.models_names[model_1], color='blue')
        plt.scatter(self.dim, f1_mean, color='blue')
        plt.semilogx(self.dim, f1_mean_1, lw=2,  label=self.models_names[model_2], color='red')
        plt.scatter(self.dim, f1_mean_1, color='red')
        plt.fill_between(self.dim, lower_bound, upper_bound, facecolor='#9B8BFF', alpha=0.3)
        plt.fill_between(self.dim, lower_bound_1, upper_bound_1, facecolor='#FF8484', alpha=0.3)
        plt.legend(loc='lower right')
        plt.ylim(0,1)
        plt.grid()
        plt.show()

        