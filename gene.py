@@ -0,0 +1,451 @@

# coding: utf-8

# In[1]:

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


# In[10]:

'''
seqFolder = '/Users/maggie/Desktop/TSS_Project_folder/scikit_learn/gene_start/seqFolder_1500bp/'
os.chdir(seqFolder)
seq_pattern = '*.test'
result_folder='/Users/maggie/Desktop/TSS_Project_folder/scikit_learn/gene_start/Result_folder/test/'

timestart = datetime.now().strftime("%Y%m%d-%H%M%S")[:-2]
#20141012-111738
report = result_folder+ 'classification_report_genestarts_'+strand+'_'+timestart+'.txt'
classification_report_out = open(report,'a')
#print (report)
print (timestart)
classification_report_out.write(timestart+'\n')


#t0 = time()

index = 0
label_dict = []
actual_label_list = []
predicted_label_list = []
for file in glob.glob(seq_pattern):
    t0 = time()
#print (file,timeNow)
    
    chrm = file.split('.')[0]
    chrmFile = seqFolder+file
    chrmFile_resulttableFile = result_folder+chrm+'_'+strand+'_'+timestart+'.table.txt'
    #print(chrmFile)
    
    if not os.path.isfile(chrmFile_resulttableFile):
        print ('START #############',chrm, strand,'####################################\n')
        print ('### readin data:',file,'###\n')
        #result_table = open(chrmFile_resulttableFile,'w')
        classification_report_out.write('START ############# '+chrm+'---'+strand+' ####################################\n')
        classification_report_out.write('### readin data: '+file+' ###\n')
        
        #import chromosme seq
        chData = pd.read_csv(chrmFile, index_col=-1,header=None,iterator=True, chunksize=1000)
        print (chData)
        
       
         
        
        chData_target = chData.index
        chData_target.values[chData_target.values == 'L1']='1'
        chData_target.values[chData_target.values == 'L2']='2'
        chData_target.values[chData_target.values == 'L3']='3'
        chData_target.values[chData_target.values == 'L4']='4'
        chData_target.values[chData_target.values == 'L5']='0'
        chData_value = chData.values
        chData_target_label = chData.index.unique()

        
        
    
        #preprocessing
        print('### preprocessing/ mean normalization ###','\n')
        classification_report_out.write('### preprocessing/ mean normalization ###'+'\n')
        chrmtest_data =scaler.transform(chData_value)
        
        #print (chData_target_label)
        print("### Predicting labels on ,",chrm,'###\n')
        classification_report_out.write("### Predicting labels on ,"+chrm+' ###\n')
        
        # predict label
        y_pred_chrm = clf_rbf.predict(chrmtest_data)

        #y_pred_chrm_prob give probablility for each instance
        y_pred_chrm_prob = clf_rbf.predict_proba(chrmtest_data)

        #output result table 
        print ('### Output result table',chrmFile_resulttableFile,'###\n' )
        classification_report_out.write('### Output result table '+chrmFile_resulttableFile+' ###\n' )
        
        result = pd.DataFrame(y_pred_chrm)
        result_prob =  pd.DataFrame(y_pred_chrm_prob)
        actual_label = pd.DataFrame(chData_target.values)
        #if y_pred_chrm[0] == 1:
         #   prob ="%.3f" % y_pred_chrm_prob[0][1]
        #elif y_pred_chrm[0] == 0:
          #  prob ="%.3f" % y_pred_chrm_prob[0][0]
        
        result['label'],result['prob0'], result['prob1'] = actual_label[0],result_prob[0], result_prob[1]
        result.columns = ['predicted_label', 'old_label','probility_for_0','probility_for_1']
        #print (y_pred_chrm[0] ,prob)
        #result['label'],result['prob0'], result['prob1'] = actual_label[0],result_prob[0], result_prob[1]
        #result['label'],result['probability'],result['chrm'] = actual_label[0],prob,chrm
        #result.columns = ['predicted_label', 'actual_label','probability','chromosome']
        result.to_csv(chrmFile_resulttableFile)
        timeUsed = "done in %0.3fs" % (time() - t0)
        #print("\ndone in %0.3fs" % (time() - t0),'\n')
        print(timeUsed+'\n')        
        print(classification_report(chData_target.values.astype(str), y_pred_chrm.astype(str) ))
        print(confusion_matrix(chData_target.values.astype(str), y_pred_chrm.astype(str)))
        
        #classification_report_out.write('####################################################################\n')
        classification_report_out.write('\n result statistic '+chrm+'--------'+strand+'--------\n')
        classification_report_out.write(classification_report(chData_target.values.astype(str), y_pred_chrm.astype(str) ))
        classification_report_out.write('\n')
        classification_report_out.write(timeUsed+'\n')
        classification_report_out.write('####################################n')
        print ('END ############',chrm, strand,'####################################\n')
        classification_report_out.write('End ############# '+chrm+'---'+strand+' ###################################\n\n')
     
     
        '''
##classification_report_out.close()


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


'''
############################3
# prediction #line by line
############################3
seqFolder = '/Users/maggie/Desktop/TSS_Project_folder/scikit_learn/gene_start/seqFolder_1500bp/'
os.chdir(seqFolder)

result_folder='/Users/maggie/Desktop/TSS_Project_folder/scikit_learn/gene_start/Result_folder/'

report = result_folder+'classification_report_'+strand+'.txt'
classification_report_out = open(report,'a')
#print (strand)
from time import time
#t0 = time()

seq_pattern = '*.seq.txt'


for file in glob.glob(seq_pattern):
    t0 = time()
    index = 0
    label_dict = []
    actual_label_list = []
    predicted_label_list = []
    print (file)
    chrm = file.split('.')[0]
    chrmFile = seqFolder+file
    chrmFile_resulttableFile = result_folder+chrm+'.'+strand+'.table.bed'
    #print(chrmFile)
    if not os.path.isfile(chrmFile_resulttableFile):
        print('creating file:',chrmFile_resulttableFile )
        result_table = open(chrmFile_resulttableFile,'w')
        with open (chrmFile, 'r') as ChrmIn:#pd.read_csv(chrmFile, index_col=-1,header=None) as chrmIn:
            for line in ChrmIn:
                index = index +1
                fi = line.strip().split(',')
                chrm_window = map(float, fi[0:1500])
            #test_window = scaler.transform(chrm_window)
                if strand == 'positive':
                    test_window = scaler.transform(chrm_window)
                elif strand == 'negative':
                    test_window = scaler.transform(chrm_window)[::-1]
            #test_window = scaler.fit(chrm_window).transform(chrm_window)
                if fi[-1] == 'L5':
                    actual_label = 0 #fi[-1].replace('5', '0')            
                else:
                    actual_label= int(fi[-1][-1])
                actual_label_list.append(actual_label)
            #prediction
                y_pred_chrm = clf_rbf.predict(test_window)
                y_pred_chrm_prob = clf_rbf.predict_proba(test_window)
                predicted_label_list.append(y_pred_chrm[0])
                if y_pred_chrm[0] == 1:
                    prob ="%.3f" % y_pred_chrm_prob[0][1]
                elif y_pred_chrm[0] == 0:
                    prob ="%.3f" % y_pred_chrm_prob[0][0]
        
                label_dict.append([chrm,index,actual_label,y_pred_chrm[0],prob,'\n'])
                labelList=[chrm,str(index),str(actual_label),str(y_pred_chrm[0]),str(prob),'\n']
                label_dict_seq = '\t'.join(labelList)
                result_table.write(label_dict_seq)
    # report 
        print ('listlen:',len(actual_label_list))
        print(classification_report(np.asarray(actual_label_list).astype(str),np.asarray(predicted_label_list).astype(str) ))
        print(confusion_matrix(np.asarray(actual_label_list).astype(str), np.asarray(predicted_label_list).astype(str)))
        classification_report_out.write(classification_report(np.asarray(actual_label_list).astype(str),np.asarray(predicted_label_list).astype(str) ))
        classification_report_out.write(confusion_matrix(np.asarray(actual_label_list).astype(str), np.asarray(predicted_label_list).astype(str)))
        classification_report_out.write('####################################################################\n')
        classification_report_out.write(chrm+'--------'+strand+'--------')
        classification_report_out.write(classification_report(np.asarray(actual_label_list).astype(str),np.asarray(predicted_label_list).astype(str) ))
        classification_report_out.write('\n')
    #classification_report_out.write(confusion_matrix(np.asarray(actual_label_list).astype(str), np.asarray(predicted_label_list).astype(str)))
        classification_report_out.write('\n')
        classification_report_out.write('####################################################################\n')
        print("done in %0.3fs" % (time() - t0))
    
        result_table.close()
    

    classification_report_out.close()
'''


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



