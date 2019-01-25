import json
import sys
from pprint import pprint
def read_file(prob_file,file_name):
    json_data=json.load(open(prob_file)) #reading the weight vector file
    #print json_data["class2"]["weight"]
    top_words=json_data["top_words"] #reading the stop words
    weight_1=json_data["class1"]["weight"] #reading the final weight vector of class1
    bias_1=json_data["class1"]["bias"] #reading the final bias of class1
    weight_2=json_data["class2"]["weight"] #reading the final weight vector of class2
    bias_2=json_data["class2"]["bias"] #reading the final bias of class2
    file_content=open(file_name,'r')
    feature=dict()
    file_write=open('percepoutput.txt','w')
    for line in file_content: #reading each line in the test file
        feature={}
        a_1=0
        a_2=0
        words=line.split()
        id=words[0]
        ignore=['.',',',"'",'"',')','(','{','}','[',']','/',';',':']
        for w in words[1:]: #calculating the feature vector for each line and ignoring the stop words
            w=w.lower()
            '''if w==":(" or w==":)":
                pass'''
            w=''.join(char for char in w if char not in ignore)
            if w!="" and w not in top_words:
                try:
                    feature[w]+=1
                except:
                    feature[w]=1
        #calculating the activation below for both the classes
        for w in feature.keys():
            try:
                a_1+=weight_1[w]*feature[w]
            except:
                pass
            try:
                a_2+=weight_2[w]*feature[w]
            except:
                pass
        a_1+=bias_1
        a_2+=bias_2
        file_write.write(str(id)+" ")
        if a_1<0:
            file_write.write("Fake"+" ")
        else:
            file_write.write("True"+" ")
        if a_2<0:
            file_write.write("Neg"+"\n")
        else:
            file_write.write("Pos"+"\n")

#read_file("averagedmodel.txt","dev-text.txt")
read_file(sys.argv[1],sys.argv[2])

