import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import re
import string
import pickle

def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W"," ",text) 
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)    
    return text

def output_lable(n):
    if n == 0:
        return "Fake News"



    elif n == 1:
        return "Not A Fake News"
    
def manual_testing(news):
    testing_news = {"text":[news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt) 
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)
    pred_LR = LR.predict(new_xv_test)
    pred_DT = DT.predict(new_xv_test)
    pred_RFC = RFC.predict(new_xv_test)

    return print("\n\nLR Prediction: {} \nDT Prediction: {} \nRFC Prediction: {}".format(output_lable(pred_LR[0]), 
                                                                                                              output_lable(pred_DT[0]),  
                                                                                                              output_lable(pred_RFC[0])))

if __name__ == "__main__":    

    df = pd.read_csv("train.csv")
    X=df.drop('label',axis=1)
    df=df.dropna()
    df = df.sample(frac = 1)
    df.reset_index(inplace = True)
    df.drop(["index"], axis = 1, inplace = True)
    df.columns
    df["text"] = df["text"].apply(wordopt)

    x = df["text"]
    y = df["label"] 
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
    vectorization = TfidfVectorizer()
    xv_train = vectorization.fit_transform(x_train)
    xv_test = vectorization.transform(x_test)

    LR = LogisticRegression()
    LR.fit(xv_train,y_train)

    pred_lr=LR.predict(xv_test)
    LR.score(xv_test, y_test)


    file = open('model.pkl', 'wb')
    pickle.dump(LR, file)
    file.close()





