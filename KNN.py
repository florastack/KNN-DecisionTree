import numpy as np
import pandas as pd                      
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import rcParams
import urllib.request
from scipy.io.arff import loadarff
from io import StringIO, BytesIO

#set the seed for reproducibility
np.random.seed(2022)

euclidean = lambda x1, x2: np.sqrt(np.sum((x1 - x2)**2,axis = -1))
manhattan = lambda x1, x2: np.sum(np.abs(x1 - x2), axis=-1)

class KNN:
    def __init__(self, K, dist_fn):
        self.dist_fn = dist_fn                                                 
        self.K = K
        return
    
    def fit(self, x, y):
        self.x = x
        self.y = y
        self.C = np.max(y) + 1
        return self
    
    def predict(self, x_test):    
        num_test = x_test.shape[0]
        
        #calculate distance between the training & test samples, need to increase the dimenstion so that
        #1.numpy could broadcast 2. to get test set element to compare with each and everyone of taining set
        #distances: shape [num_test, num_train]                      
        distances = self.dist_fn(self.x[None,:,:], x_test[:,None,:])  
        knns = np.zeros((num_test, self.K), dtype=int)
        y_prob = np.zeros((num_test, self.C))
        for i in range(num_test):
            knns[i,:] = np.argsort(distances[i])[:self.K]                  #returns the index of the sorted element and only stores first K ones
            y_prob[i,:] = np.bincount(self.y[knns[i,:]], minlength=self.C) #counts the number of instances of each class in the K-closest training samples
        y_prob /= self.K                                                   #divided by K so that scale the value from zero to 1.
        return y_prob, knns
    
    # evaluate accuracy
    def evaluate_acc(self, y_pred, y_true):
        true_pos = len(np.argwhere((y_pred ==0) & (y_true == 0)))
        true_neg = len(np.argwhere((y_pred ==1) & (y_true == 1)))        
        false_pos = len(np.argwhere((y_pred ==0) & (y_true == 1)))
        false_neg = len(np.argwhere((y_pred ==1) & (y_true == 0)))
        
        return (true_pos + true_neg)/(true_pos + true_neg + false_pos + false_neg)
    
def plot_correlation(data):
    rcParams['figure.figsize'] = 15, 20
    fig = plt.figure()
    sns.heatmap(data.corr(), annot=True, fmt=".2f")
    plt.show()
    fig.savefig('corr.png')    

dist_fns = [euclidean, manhattan]
dist_fns_names = ["euclidean", "manhattan"]

# run for diabetic data
diabetic_uci_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00329/messidor_features.arff'
resp = urllib.request.urlopen(diabetic_uci_url)
data_arff, meta = loadarff(StringIO(resp.read().decode('utf-8')))
names = ["quality", "pre_screening", "MA_2", "MA_3", "MA_4", "MA_5", "MA_6", "MA_7",
         "exudates_8", "exudates_9", "exudates_10", "exudates_11", "exudates_12",
         "exudates_13", "exudates_14", "exudates_15", "distance", "diameter",
         "AM_FM", "Class"]

diab_df= pd.DataFrame(data_arff)
missingdata_df = diab_df[diab_df.eq('?').any(1)] #df with rows that have missing data
diab_df = pd.concat([diab_df, missingdata_df]).drop_duplicates(keep=False) #drop the duplicated df
for col in diab_df.columns:
  diab_df[col] = diab_df[col].astype('float')
diab_df .columns = names
first_column = diab_df.pop('Class')
diab_df.insert(0, 'Class', first_column)
data = diab_df.values

# get some vis on data
plot_correlation (diab_df)
x, y = data[:,1:].astype(float) , data[:,0].astype(int)                            
(N,D), C = x.shape, np.max(y)+1                   
inds = np.random.permutation(N)  #generates an indices array from 0 to N-1 and permutes it 

#split the dataset into train60%, validation20% and test20%
x_train, y_train = x[inds[:int(0.6*N)]], y[inds[:int(0.6*N)]]
x_validate, y_validate = x[inds[int(0.6*N):int(0.8*N)]], y[inds[int(0.6*N):int(0.8*N)]]
x_test, y_test = x[inds[int(0.8*N):]], y[inds[int(0.8*N):]]

results = []
for (dist_fn, dist_fn_name) in zip(dist_fns, dist_fns_names ):
    for k in range(1, 21):
          model = KNN(k, dist_fn)
          model.fit(x_train, y_train)
          y_prob_validate, knns_validate = model.predict(x_validate)
        
          y_pred_validate = np.argmax(y_prob_validate,axis=-1) 
          accuracy = model.evaluate_acc(y_pred_validate, y_validate)

          results.append([dist_fn_name, k, accuracy])

df_diatresult = pd.DataFrame(data = results, columns = ["DistanceFN", "K", "Accuracy"])

k_best_diat = df_diatresult.at[df_diatresult["Accuracy"].idxmax(),"K"]
cosfFN_best_diat = df_diatresult.at[df_diatresult["Accuracy"].idxmax(),"DistanceFN"]

model_test  = KNN(1, manhattan)
model_test.fit(x_train, y_train)
y_prob_test, knns_validate = model.predict(x_test)
        
y_pred_test= np.argmax(y_prob_test,axis=-1) 
diat_accuracy_test = model.evaluate_acc(y_pred_test, y_test)

# run for hep data
hep_df = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/hepatitis/hepatitis.data", header=None)
missingdata_df = hep_df[hep_df.eq('?').any(1)] #df with rows that have missing data
hep_df = pd.concat([hep_df, missingdata_df]).drop_duplicates(keep=False) #drop the duplicated df
for col in hep_df.columns:
  hep_df[col] = hep_df[col].astype('float')
hep_df.columns = ['class','age', 'sex', 'steriod', 'antivirals',
'fatigue', 'malaise', 'anorexia','liver_big', 'liver_firm', 'spleen_palpable', 
'spiders', 'ascites', 'varices', 'blirubin', 'alk', 'sgot','albumin', 'potime', 'hitoloty']

hep_df = hep_df.replace({'class': {1: 0, 2: 1}})
    
data = hep_df.values

x, y = data[:,1:], data[:,0].astype(int)                              

(N,D), C = x.shape, np.max(y)+1                   

inds = np.random.permutation(N)  #generates an indices array from 0 to N-1 and permutes it 

#split the dataset into train60%, validation20% and test20%
x_train, y_train = x[inds[:int(0.6*N)]], y[inds[:int(0.6*N)]]
x_validate, y_validate = x[inds[int(0.6*N):int(0.8*N)]], y[inds[int(0.6*N):int(0.8*N)]]
x_test, y_test = x[inds[int(0.8*N):]], y[inds[int(0.8*N):]]
results = []

for (dist_fn, dist_fn_name) in zip(dist_fns, dist_fns_names ):
    for k in range(1, 21):
        model = KNN(k, dist_fn)
        model.fit(x_train, y_train)
        y_prob_validate, knns_validate = model.predict(x_validate)
        
        y_pred_validate = np.argmax(y_prob_validate,axis=-1) 
        accuracy = model.evaluate_acc(y_pred_validate, y_validate)

        results.append([dist_fn_name, k, accuracy])

df_hepresult = pd.DataFrame(data = results, columns = ["DistanceFN", "K", "Accuracy"])

k_best_hep = df_hepresult.at[df_hepresult["Accuracy"].idxmax(),"K"]
distanceFN_best_hep = locals()[df_hepresult.at[df_hepresult["Accuracy"].idxmax(),"DistanceFN"]]

model_test  = KNN(7, manhattan)
model_test.fit(x_train, y_train)
y_prob_test, knns_validate = model.predict(x_test)
        
y_pred_test= np.argmax(y_prob_test,axis=-1) 
hep_accuracy_test = model.evaluate_acc(y_pred_test, y_test)