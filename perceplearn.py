import json
import sys
import operator
import math
from random import shuffle

#The function below computes the feature vector for every sentence 
def get_feature_vectors(file_name,feat_vec,doc_words):
    file_content=open(file_name,'r')
    for line in file_content:
        feature={}
        ignore=['.',',',"'",'"',')','(','{','}','[',']','/',';',':']
        words=line.split()
        id=words[0] #unique id of every document
        class1=words[1] #reads if it's a Fake/True class
        class2=words[2] #reads if it's a Pos/Neg class
        line_words=words[3:] #the rest of the document
        for w in line_words: # for each line calculate the counts of each vector and keep track of all the word counts in the corpus in doc_words
            w=w.lower()
            '''if w==":(" or w==":)":
                pass'''
            w=''.join(char for char in w if char not in ignore)
            if w!="":
                try:
                    feature[w]+=1
                except:
                    feature[w]=1
                try:
                    doc_words[w]+=1
                except:
                    doc_words[w]=1
        feat_vec[id]={"class1":class1,"class2":class2,"feat_vec":feature}

#
def read_file(file_name):
    feat_vec={}
    doc_words={}
    get_feature_vectors(file_name,feat_vec,doc_words)
    threshold=int(0.00111958*len(doc_words.keys()))
    my_top=dict(sorted(doc_words.iteritems(), key=operator.itemgetter(1), reverse=True)[:threshold])
    my_top=my_top.keys() #calculating the stop words
    weight_1={} #weight vector of class1
    weight_2={} #weight vector of class2
    bias_1=0 
    bias_2=0
    avg_weight_1={} #average weight of class1
    avg_weight_2={} #average weight of class2
    avg_bias_1=0
    avg_bias_2=0
    max_itr=75
    c1={"Fake":-1,"True":1}
    c2={"Neg":-1,"Pos":1}
    i=0
    shuffled_feat=feat_vec.keys()
    count=1 #initializing the count
    while i<max_itr: # in every iteration the input is given in different order
        for id in shuffled_feat: 
            my_data=feat_vec[id]
            class1=my_data["class1"]
            class2=my_data["class2"]
            feature=my_data["feat_vec"]
            a_1=0 #activation a_1 for class1
            a_2=0 # activation a_2 foor class2
            for w in feature.keys(): #computes the activations a_1 and a_2 for the current sentance
                try:
                    a_1+=weight_1[w]*feature[w]
                except:
                    weight_1[w]=0
                try:
                    a_2+=weight_2[w]*feature[w]
                except:
                    weight_2[w]=0
            a_1+=bias_1
            a_2+=bias_2
            y_1=c1[class1]
            y_2=c2[class2]
            if y_1*a_1<=0:
                for w in feature.keys(): #updating weights and bias of class1 
                    weight_1[w]+=y_1*feature[w] 
                    try:
                        avg_weight_1[w]+=y_1*feature[w]*count
                    except:
                        avg_weight_1[w]=y_1*feature[w]*count
                bias_1+=y_1
                avg_bias_1+=y_1*count
            if y_2*a_2<=0: #updating weights and bias of class2
                for w in feature.keys():
                    weight_2[w]+=y_2*feature[w]
                    try:
                        avg_weight_2[w]+=y_2*feature[w]*count
                    except:
                        avg_weight_2[w]=y_2*feature[w]*count
                bias_2+=y_2
                avg_bias_2+=y_2*count
            count+=1
        i+=1
        #shuffle(shuffled_feat)
    average_wt1={}
    average_wt2={}
    average_bias1=bias_1-(1/float(count))*avg_bias_1 #average bias of class 1
    average_bias2=bias_2-(1/float(count))*avg_bias_2 #average bias of class 2
    #print len(weight_1.keys())
    #print len(weight_2.keys())
    #computes the average weights of classes
    for w in weight_1.keys():
        try:
            average_wt1[w]=weight_1[w]-(1/float(count))*avg_weight_1[w]
        except:
            average_wt1[w]=weight_1[w]
    for w in weight_2.keys():
        try:
            average_wt2[w]=weight_2[w]-(1/float(count))*avg_weight_2[w]
        except:
            average_wt2[w]=weight_2[w]
    final_write={"class1":{"weight":weight_1,"bias":bias_1},"class2":{"weight":weight_2,"bias":bias_2},"top_words":my_top}
    avg_write={"class1":{"weight":average_wt1,"bias":average_bias1},"class2":{"weight":average_wt2,"bias":average_bias2},"top_words":my_top}
        #file_write_test=open("test.txt",'w')
        #file_write_test.write(str(final_write))
    file_write=open('vanillamodel.txt','w') #writes the final weight vectors and the stops words of class1 and class2 to a file
    file_avg_write=open('averagedmodel.txt','w') #writes the final average weight vectors and stop words of class1 and class2 to a file
    json.dump(avg_write,file_avg_write)
    json.dump(final_write, file_write)


read_file(sys.argv[1])
#read_file('train-labeled.txt')
