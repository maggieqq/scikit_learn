from __future__ import print_function
#'positive'
#strand = 'negative'
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
import numpy as np
import csv
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
import sys
from pandas import DataFrame, read_csv
# Basic CSV IO
import numpy as np
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn import svm
from time import time
from sklearn.cross_validation import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import RandomizedPCA
from sklearn.svm import SVC
from collections import defaultdict
import csv,sys,os,random,math,array,glob,re,time,timeit
import logging
import matplotlib.pyplot as plt
from time import time
from datetime import datetime

#strand = sys.argv[1]
strand='negative'

chrm_len = {'Pf3D7_01_v3': 640851,'Pf3D7_02_v3': 947102, 
'Pf3D7_03_v3': 1067971,'Pf3D7_04_v3': 1200490,'Pf3D7_05_v3': 1343557,
'Pf3D7_06_v3': 1418242,  'Pf3D7_07_v3': 1445207, 'Pf3D7_08_v3': 1472805,   
'Pf3D7_09_v3': 1541735,'Pf3D7_10_v3': 1687656,'Pf3D7_11_v3': 2038340, 
'Pf3D7_12_v3': 2271494,'Pf3D7_13_v3': 2925236,  'Pf3D7_14_v3': 3291936,
}

def read_data(file_name):
    f = open(file_name)
    #ignore header
    f.readline()
    samples = []
    target = []
    for line in f:
        line = line.strip().split(",")
        sample = [float(x) for x in line]
        samples.append(sample)
    return samples

def write_delimited_file(file_path, data,header=None, delimiter=","):
    f_out = open(file_path,"w")
    if header is not None:
        f_out.write(delimiter.join(header) + "\n")
    for line in data:
        if isinstance(line, str):
            f_out.write(line + "\n")
        else:
            f_out.write(delimiter.join(line) + "\n")
    f_out.close()

def read_csv(file_path, has_header = True):
    with open(file_path) as f:
        if has_header: f.readline()
        data = []
        for line in f:
            line = line.strip().split(",")
            data.append([float(x) for x in line])
    return data

def write_csv(file_path, data):
    with open(file_path,"w") as f:
        for line in data: f.write(",".join(line) + "\n")
def moving_average(a, n) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


# find consecutive group > 50
def group_consecutives(vals, step=1):
    """Return list of consecutive lists of numbers from vals (number list)."""
    run = []
    result = [run]
    expect = None
    for v in vals:
        if (v == expect) or (expect is None):
            run.append(v)
        else:
            run = [v]
            result.append(run)
        expect = v + step
    return result
############################3
# import data #####
#################
if strand == 'positive':
    File = '/Users/maggie/Desktop/TSS_Project_folder/Gene_v11/coordinateFiles/gene_start/1500bp/MargSize_50/sum_sequence/Positive_1500bp_gene_start_50_trainset_6000.seq.txt'
elif strand == 'negative': 
    File = '/Users/maggie/Desktop/TSS_Project_folder/Gene_v11/coordinateFiles/gene_start/1500bp/MargSize_50/sum_sequence/Negative_1500bp_gene_start_50_trainset_6000.seq.txt'
#Data = pd.read_csv(PosFile, index_col=-1,header=None,sep = ',')

Data = pd.read_csv(File, index_col=-1,header=None,sep = ',')
Data_label = Data.index
Data_label.values[Data_label.values == 5]='0'
Data_value = Data.values
Data_target_label = Data.index.unique()
scaler = preprocessing.StandardScaler().fit(Data_value)

# No Normalization at all
train_data = scaler.transform(Data_value)
print (Data_target_label)
############################3
#trian for model
############################3
print('trianing model using file: ', File)
print('strand:',strand)
clf_rbf = svm.SVC(kernel = 'rbf', C=10, gamma=0.001, probability=True) 
clf_rbf.fit(train_data, Data_label)
all_score = clf_rbf.score(train_data, Data_label)
#y_pred_chrm = clf_rbf.predict(train_data)
#y_pred_chrm_prob = clf_rbf.predict_proba(train_data)
#Data_label=map(int, Data_label)
#Data_label = np.asarray(Data_label)
#print(classification_report(Data_label.astype(str), y_pred_chrm.astype(str) ))
#print(confusion_matrix(Data_label.astype(str), y_pred_chrm.astype(str)))
print (all_score,'\n') #0.987
print ('##################################################################')


############################3
# prediction #chunk by chunk
############################3
from time import time
from pandas import *
seqFolder = '/Users/maggie/Desktop/TSS_Project_folder/scikit_learn/gene_start/seqFolder_1500bp/'
os.chdir(seqFolder)

result_folder='/Users/maggie/Desktop/TSS_Project_folder/scikit_learn/gene_start/Result_folder/'
timestart = datetime.now().strftime("%Y%m%d-%H%M%S")[:-2]
#20141012-111738
report = result_folder+ 'classification_report_genestarts_'+strand+'_'+timestart+'.txt'
classification_report_out = open(report,'a')
#print (strand)

seq_pattern = 'Pf3D7_02_v3.label.txt.seq.replaced.txt'
result_array=['actual-label','predicted-label','score-0','score-1']

for file in glob.glob(seq_pattern):
    t0 = time()
    index = 0
    label_dict = []
    actual_label_list = []
    predicted_label_list = []
    overall_list=[]
    print (file)
    chrm = file.split('.')[0]
    chrmFile = seqFolder+file
    chrmFile_resulttableFile = result_folder+chrm+'_'+strand+'_'+timestart+'.table.txt'
    print(chrmFile)

    if not os.path.isfile(chrmFile_resulttableFile):
        print ('START #############',chrm, strand,'####################################\n')
        print ('### readin data:',file,'###\n')
        result_table = open(chrmFile_resulttableFile,'a')
        classification_report_out.write('START ############# '+chrm+'---'+strand+' ####################################\n')
        classification_report_out.write('### readin data: '+file+' ###\n')
	count = 0
        #import chromosme seq
        chData_file = pd.read_csv(chrmFile, index_col=-1,header=None,iterator=True, chunksize=500000)
        for chData in chData_file:
            chData_target = chData.index
            chData_value = chData.values
            chData_target_label = chData.index.unique()
            timeUsed = "done in %0.3fs" % (time() - t0)
	    count=count+1
            print(timeUsed+'\n',count)
            if strand == 'positive':
                test_window = scaler.transform(chData_value)
            elif strand == 'negative':
                test_window = scaler.transform(chData_value[:,::-1])
                
            
            
            y_pred_chrm = clf_rbf.predict(test_window)
            y_pred_chrm_prob = clf_rbf.predict_proba(test_window)
            
            columnbind =np.column_stack((chData_target,y_pred_chrm,y_pred_chrm_prob))  
            result_array= np.vstack([result_array, columnbind])
            #print(result_array)
            actual_label_list.extend(np.asarray(chData_target))
            predicted_label_list.extend(np.asarray(y_pred_chrm))
            
        result = pd.DataFrame(result_array[1:])
        result.to_csv(chrmFile_resulttableFile)
        print(classification_report(np.asarray(actual_label_list).astype(str),np.asarray(predicted_label_list).astype(str) ))
        print(confusion_matrix(np.asarray(actual_label_list).astype(str), np.asarray(predicted_label_list).astype(str)))

            
        #classification_report_out.write('####################################################################\n')
        timeUsed = "done in %0.3fs" % (time() - t0)
        classification_report_out.write('\n result statistic '+chrm+'--------'+strand+'--------\n')
        classification_report_out.write(classification_report(np.asarray(actual_label_list).astype(str),np.asarray(predicted_label_list).astype(str) ))
        classification_report_out.write('\n')
        classification_report_out.write(timeUsed+'\n')
        classification_report_out.write('####################################n')
        print ('END ############',chrm, strand,'####################################\n')
        classification_report_out.write('End ############# '+chrm+'---'+strand+' ###################################\n\n')
                
            
            
classification_report_out.close()  
