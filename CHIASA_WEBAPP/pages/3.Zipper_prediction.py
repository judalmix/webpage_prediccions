#imports

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
import pickle
import numpy
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from xgboost.sklearn import XGBClassifier
import shap
from streamlit_shap import st_shap



hide = """
<style>
div[data-testid="stConnectionStatus"] {
    display: none !important;
</style>
"""
st.markdown(hide, unsafe_allow_html=True)



#functions
def load_model(filename): #load the model from notebook
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return data


def apply_regression(data,nom_ultima_col):
    data=data.drop('Quartils',axis=1)
    model=LinearRegression()
    y=data.loc[: , nom_ultima_col]
    x=data.loc[: , data.drop(nom_ultima_col, axis=1).columns]
    X_train, X_test, Y_train, Y_test= train_test_split(x,y,train_size=0.8)
    regression_model=model

    model_fit=regression_model.fit(X_train, Y_train)
    predict = model_fit.predict(X_test)
    relative_error=np.abs(predict-Y_test)/np.abs(Y_test)
    relative_error_mean=relative_error.mean()

    return regression_model, X_test, x, Y_test,relative_error_mean

def multiclass_classification(data,nom_ultima_col):
    data=data.drop(nom_ultima_col,axis=1)
    model_xgboost=XGBClassifier()
    y2=data.loc[: , 'Quartils']
    x2=data.loc[: , data.drop('Quartils', axis=1).columns]
    X_train2, X_test2, Y_train2, Y_test2= train_test_split(x2,y2,train_size=0.8, test_size=0.2)
    model_xgboost_multiclass = model_xgboost.fit(X_train2, Y_train2)
    predict_xgb = model_xgboost_multiclass.predict(X_test2)
    
    return  model_xgboost,X_train2, X_test2, Y_train2, Y_test2,x2,y2

def array_to_dataset(idx,num_cols,new_df):
    new_data=pd.DataFrame()
    j=0
    for i in new_df.columns: 
        if j<num_cols:
            new_data[i]=[idx[j]]
            new_data.i=new_data.astype(int)
        j=j+1
    return new_data

def insert_zipper(idx,dataframe):
    inputs=[]
    columnes=dataframe.columns
    for i in range(dataframe.shape[1]):
        if columnes[i]=='Familia' or columnes[i]=='Stopers' or columnes[i]=='Sliders' or columnes[i]=='Teeth' or columnes[i]=='Color' or columnes[i]=='Llargada' or columnes[i]=='Label' :
            keys=str(idx+1)+ str(i+1)
            #input=st.number_input()
            text=st.text_input('Insert the'+ columnes[i]+ 'Code:',key=keys)
            hide = """<style>div[data-testid="stConnectionStatus"] {    display: none !important;</style>"""
            st.markdown(hide, unsafe_allow_html=True)
            inputs.append(text)


    df=pd.DataFrame([inputs], columns=['Familia', 'Stopers', 'Sliders', 'Teeth', 'Color','Label', 'Llargada'])    


    return df



def zippers_model(new_df, df_not_encoded,multiclass_model, regression_model, X_test_regression_reduced,x_regression_reduced, relative_error_mean_reduced,nom_ultima_col,quartils):
    search_zipper=insert_zipper(-1,dataset)
    finish_zipper=st.button('Submit Zipper')

    if finish_zipper: 
        counter=0
        for i in range(len(df_not_encoded)):
            if (df_not_encoded['Familia'].iloc[i]==search_zipper.iloc[0][0]) and (df_not_encoded['Stopers'].iloc[i]==search_zipper.iloc[0][1])and (df_not_encoded['Sliders'].iloc[i]==search_zipper.iloc[0][2]) and (df_not_encoded['Teeth'].iloc[i]==search_zipper.iloc[0][3])and(df_not_encoded['Color'].iloc[i]==search_zipper.iloc[0][4]) and (df_not_encoded['Label'].iloc[i]==search_zipper.iloc[0][5]) and (df_not_encoded['Llargada'].iloc[i]==search_zipper.iloc[0][6]):
                counter=i
                st.write('The zipper exists in the data!')
                st.write('The zipper introduced is: ', search_zipper)
                
            
    
        counter=int(counter)
        aux=new_df.iloc[counter]
        aux1=aux.drop(['Quartils',nom_ultima_col])
        new_data=new_data=array_to_dataset(aux1,len(aux1),new_df)

        tab1,tab2= st.tabs(["MULTICLASS prediction", "REGRESSION prediction"])
        with tab1: 
            new_prediction_multiclass= multiclass_model.predict(new_data)
            if new_prediction_multiclass==0: 
                st.write('The zipper will be sold around this quantity: ', 0 ,'-',quartils[0])
            elif new_prediction_multiclass==1:
                st.write('The zipper will be sold around this quantity: ', quartils[0],'-',quartils[1])
            elif new_prediction_multiclass==2: 
                st.write('The zipper will be sold around this quantity: ', quartils[1] ,'-',quartils[2])
            else: 
                st.write('The zipper will be sold around this quantity: ', quartils[2] ,'-',quartils[3])
        with tab2: 
            st.write('')
            results_linear_regression = regression_model.predict(new_data) 
            relative_error=((np.abs(results_linear_regression-aux[nom_ultima_col]))/(np.abs(aux[nom_ultima_col])))*100
            if relative_error>=relative_error_mean_reduced:
                st.write('The linear regression is not giving an acurate prediction for this zipper.')
                st.write('Is better to rely on the MULTICLASS CLASSIFICATION.')
            else:
                st.write('The prediction of the quantity of zippers that will be sold in the target variable is: ')
                st.write(truncate(results_linear_regression[0],0))
        
        
            
    
def truncate(number, max_decimals):
    int_part, dec_part = str(number).split(".")
    return float(".".join((int_part, dec_part[:max_decimals])))

def model(df,new_df,regression_model_reduced,values_dict,num,nom_ultima_col,relative_error_mean_reduced,multiclass_model):
    num=str(num)
    new_df=new_df.drop('Quartils',axis=1)
    target=new_df.iloc[:,7:]
    not_target=new_df.iloc[:,:7]
    #model=LinearRegression()
    model_fit=regression_model_reduced.fit(not_target, target)
    prediction=model_fit.predict(not_target)
    for i in range(prediction.shape[0]):
        for j in range(prediction.shape[1]):
            if prediction[i][j]<=0:
                prediction[i][j]=0
            else:
                prediction[i][j]=truncate(prediction[i][j],0)

    new_future_sales=pd.DataFrame(prediction)
    not_target = not_target.reset_index(drop=True)
    result = pd.concat([not_target, new_future_sales],axis=1)
    new_cols = {}
    for i, col in enumerate(result.columns[7:]):
        new_col_name = f'{i+1}{"  Predicció" if i%10==0 and i!=10 else ""} dels següents' + num + 'mesos'
        new_cols[col] = new_col_name 
    df_encoded = result.rename(columns=new_cols)
    df_encoded_0=df_encoded.copy()

    decodings = {}
    
    for col in df_encoded.columns[:7]:
        if col in values_dict:
            categories, _ = values_dict[col]
            decodings[col] = dict(enumerate(_))
            df_encoded[col] = df_encoded[col].map(lambda x: decodings[col][x])

    return  df_encoded_0,df_encoded

def convert_to_string(dataframe):
    num_initial_cols = 7
    # Get the number of columns that are integers
    num_int_cols = dataframe.shape[1] - num_initial_cols
    # Create a list of new column names that are strings
    new_col_names = [str(i+num_initial_cols) for i in range(num_int_cols)]
    # Rename the integer columns using the new column names
    dataframe.columns.values[num_initial_cols:num_initial_cols + num_int_cols] = new_col_names
    for i in range(len(new_col_names)):
        df=dataframe.rename(columns={num_initial_cols:new_col_names[0]})
        num_initial_cols=num_initial_cols+1
    return df
def convert_df(df):
   return df.to_csv().encode('utf-8')




dataset=st.session_state.data
num=st.session_state.numero
df_not_encoded=st.session_state.data_not_encoded
values_dict=st.session_state.diccionari


#starting plotting
st.title('Zipper prediction with ML')
st.write('')
st.write("We are going to see some predictions of our datasets using Machine Learning and Shap plots")
st.write('You will be able to see do two different types of prediction:')
st.write('1-You will be able to see the prediction of one zipper for the next month.')
st.write('2-Prediction for all the data. ')
st.write('')
st.write('')
st.write('Here you can find the prediction and the explainability of the model. ')

df=convert_to_string(dataset)
nom_ultima_col = df.columns[-1]
new_df= df[(df[nom_ultima_col]>0)]
quartils= numpy.quantile(new_df[nom_ultima_col], [0.25,0.5,0.75,1])
q=[]
for i in range(len(df)):
    if df[nom_ultima_col].iloc[i]<=quartils[0]:
        q.append(0)
    elif df[nom_ultima_col].iloc[i]>quartils[0] and df[nom_ultima_col].iloc[i]<=quartils[1]:
        q.append(1)
    elif df[nom_ultima_col].iloc[i]>quartils[1] and  df[nom_ultima_col].iloc[i]<=quartils[2]:
        q.append(2)
    else:
        q.append(3)
df['Quartils']=q




model_regression,X_test_regression_reduced,x_regression_reduced,Y_test_regression_reduced, relative_error_mean_reduced=apply_regression(df,nom_ultima_col)
model_xgboost,X_train2, X_test2, Y_train2, Y_test2,x2,y2=multiclass_classification(df,nom_ultima_col)



with st.expander("Prediction"):
       
    tab1,tab2= st.tabs(["Prediction of a zipper already in the dataset", " Prediction Dataset for the next months"])
    with tab1: 
        st.write('You will do a prediction with a zipper already in the dataset!')
        zippers_model(df,df_not_encoded, model_xgboost, model_regression, X_test_regression_reduced,x_regression_reduced, relative_error_mean_reduced,nom_ultima_col,quartils)
        
    with tab2: 
        st.write('Prediction for the next months: ')
        st.write('')
        df_encoded,df_not_encoded=model(df,df, LinearRegression(),values_dict,num,nom_ultima_col,relative_error_mean_reduced,model_xgboost)
        st.write('The prediction of the data is: ')
        st.write(df_not_encoded)
        csv = convert_df(df_not_encoded)
        st.download_button(label="Download data as CSV",data=csv,file_name='predictions.csv',mime='text/csv')

