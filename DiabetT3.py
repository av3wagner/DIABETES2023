from  PIL import Image
import numpy as np
import base64
import sys
import inspect, os
import pathlib
from pathlib import Path
from os import listdir
from os.path import isfile, join
import streamlit.components.v1 as components

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_curve
from sklearn.metrics import RocCurveDisplay

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

#logo = Image.open('FlagKZ.JPG')
options = "Home"
path = os.getcwd()
profile  = Image.open('AWagner.JPG')
profile2 = Image.open('AWprofil.jpg')
AWbrigade = Image.open('AWbrigade.jpg')

def main():
    st.set_page_config(
    page_title="Kazakhstan Diabet-Projekt",
    page_icon="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/72x72/1f440.png",
    layout="centered",
    #layout="wide",    
    initial_sidebar_state="auto",
    menu_items={
        'Get Help': 'https://github.com/foo',
        'Report a bug': "https://github.com/foo",
        'About': "Foo"
    }
)
    #st.markdown("")
    #st.image(logo, width=100 )
    #st.markdown("")
    
    st.title("Predicting Diabetes Web App")
    st.title("Please choose an option:")
    options = st.sidebar.radio('', ['Show raw data', 'over us', 'Example 18 models', 'clear'])
    if options == 'Show raw data':
        st.subheader("Diabetes Raw Dataset")
        st.write(df)  
    elif options == 'over us':
        welcome() 
    elif options == 'Example 18 models':
        st.header("Example: 18 machine learning models")
        HtmlFile = open("A++Nostalgi08.html", 'r', encoding='utf-8')
        source_code = HtmlFile.read() 
        components.html(source_code, width=900, height=2000, scrolling=True)
    elif options == 'clear':
        st.header("")
          
    st.sidebar.title("Model Selection Panel")
    st.markdown("Affected by Diabetes or not ?")
    st.sidebar.markdown("Choose your model and its parameters")
    
    @st.cache(persist = True) 
    def load_data():
        df = pd.read_csv("data/diabetes.csv")
        return df
        
    st.sidebar.subheader("Select your Classifier")
    classifier = st.sidebar.selectbox("Classifier", ("Decision Tree", "Logistic Regression", "Random Forest"))
 
    if classifier == 'Decision Tree':
        st.sidebar.subheader("Model parameters")
        criterion= st.sidebar.radio("Criterion(measures the quality of split)", ("gini", "entropy"), key='criterion')
        splitter = st.sidebar.radio("Splitter (How to split at each node?)", ("best", "random"), key='splitter')
        metrics = st.sidebar.multiselect("Select your metrics : ", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))        
        
        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Decision Tree Results")
            model = DecisionTreeClassifier(criterion=criterion, splitter=splitter)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy.round(2)*100,"%")
            st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))

            st.subheader("Confusion Matrix")
            confusion_ma(y_test, y_pred, class_names)
            
            st.subheader("ROC Curve")
            svc_disp = RocCurveDisplay.from_estimator(model, x_test, y_test)
            st.pyplot()
   
    if classifier == 'Logistic Regression':
        st.sidebar.subheader("Model Parameters")
        C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C_LR')
        max_iter = st.sidebar.slider("Maximum number of iterations", 100, 500, key='max_iter')
        #metrics = st.sidebar.multiselect("Select your metrics?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Logistic Regression Results")
            model = LogisticRegression(C=C, penalty='l2', max_iter=300) #max_iter)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy.round(2)*100,"%")
            st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
            
            st.subheader("Confusion Matrix")
            confusion_ma(y_test, y_pred, class_names)
            
            st.subheader("ROC Curve")
            svc_disp = RocCurveDisplay.from_estimator(model, x_test, y_test)
            st.pyplot()
            
    if classifier == 'Random Forest':
        st.sidebar.subheader("Model Hyperparameters")
        n_estimators = 3000
        n_estimators = st.sidebar.number_input("The number of trees in the forest", 1000, 5000, step=100, key='n_estimators')
        max_depth=19
        max_depth = st.sidebar.number_input("The maximum depth of the tree", 1, 20, step=1, key='max_depth')
        #bootstrap='True'
        #bootstrap = st.sidebar.radio("Bootstrap samples when building trees", ('True', 'False'), key='bootstrap')
        metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Random Forest Results")
            #model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap, n_jobs=-1)
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth) 
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy.round(2)*100,"%")
            st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
                       
            st.subheader("Confusion Matrix")
            confusion_ma(y_test, y_pred, class_names)
            
            st.subheader("ROC Curve")
            svc_disp = RocCurveDisplay.from_estimator(model, x_test, y_test)
            st.pyplot() 

def split(df):
    req_cols = ['Pregnancies', 'Insulin', 'BMI', 'Age','Glucose','BloodPressure','DiabetesPedigreeFunction']
    x = df[req_cols] 
    y = df.Outcome
    x = df.drop(columns=['Outcome'])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    return x_train, x_test, y_train, y_test

def confusion_ma(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot() 
    st.pyplot()

def plot_metrics(metrics_list):
    print(metrics_list)
    if 'Confusion Matrix' in metrics_list:
        st.subheader("Confusion Matrix")
        plot_confusion_matrix(model, x_test, y_test, display_labels=class_names)
        st.pyplot()
      
    if 'ROC Curve' in metrics_list:
        st.subheader("ROC Curve")
        plot_roc_curve(model, x_test, y_test)
        st.pyplot()
        
    if 'Precision-Recall Curve' in metrics_list:
        st.subheader('Precision-Recall Curve')
        plot_precision_recall_curve(model, x_test, y_test)
        st.pyplot()  
        
def load_data():
    df = pd.read_csv("data/diabetes.csv")
    return df

def read_markdown_file(markdown_file):
    return Path(markdown_file).read_text()

def welcome():
    col1, col2, col3 = st.columns( [1, 8, 1])
      
    with col1:              
        st.markdown("")
    with col2:  
        st.title("Dr. Alexander Wagner, Berlin")
        st.markdown("")
        st.markdown("")
        st.image(profile, width=400 )
        title1 = '<p style="font-family:sans-serif; color:Black; font-weight:bold; font-size: 12px;">MSD SHARP & DOHME GMBH: Alexander Wagner. Мюнхен, 2010</p>'
        st.markdown(title1, unsafe_allow_html=True)
        
        st.markdown("")
        st.image(AWbrigade, width=400 )
        new_title = '<p style="font-family:sans-serif; color:Black; font-weight:bold; font-size: 12px;">Baustelle: Alexander Wagner 4. von links. Almaty, April 1973</p>'
        st.markdown(new_title, unsafe_allow_html=True)
        
        st.markdown("")
        intro_markdown=open("DrWagnerRU.md", encoding="utf-8").read()
        st.markdown(intro_markdown, unsafe_allow_html=True)
    with col3:              
        st.markdown(""" <style> .font {
        font-size:10px ; font-family: 'Cooper Black'; color: #FF9633;} 
        </style> """, unsafe_allow_html=True)
        st.markdown('<p class="font"> </p>', unsafe_allow_html=True)

df=load_data()    
class_names = ['Diabetec', 'Non-Diabetic']
x_train, x_test, y_train, y_test = split(df)

if __name__ == '__main__':
    main()
