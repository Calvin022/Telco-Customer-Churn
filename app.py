import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics 
from xgboost.sklearn import XGBClassifier
import pickle
import streamlit as st

# loading data from csv file and selecting relevant columns
@st.cache(allow_output_mutation=True)
def load_data():
    df = pd.read_csv('Telco-Customer-Churn.csv')
    data = df[['gender','SeniorCitizen','InternetService','Contract','PaymentMethod','tenure','MonthlyCharges','Churn']]
    return data


# preprocessing the data
@st.cache(allow_output_mutation=True)
def preprocess(data):
    label_encoder = LabelEncoder()
    churn_encoder = LabelEncoder()
    churn = np.array(data['Churn'])
    data['Churn'] = churn_encoder.fit_transform(churn)
    for column in data.select_dtypes(include='object').columns:
        values = np.array(data[column])
        data[column] = label_encoder.fit_transform(values)

    X = data.drop(['Churn'], axis=1)
    y = data['Churn']
    Xtrain,Xtest,ytrain,ytest = train_test_split(X,y,test_size=0.2, random_state=0)
    return Xtrain,Xtest,ytrain,ytest,churn_encoder


# accepting user input for predictions
def accept_user_data():
    gender = st.selectbox('Gender',('Male','Female'))
    seniorcitizen = st.selectbox('SeniorCitizen',('Yes','No'))
    internetService = st.selectbox('Internet Service',('Fiber Optic','DSL','No Internet Service'))
    contract = st.selectbox('Contract',('Month-to-month','Two year', 'One year'))
    paymentMethod = st.selectbox('Payment Method',('Electronic Cheque','Mailed Cheque','Bank Transfer (Automatic)','Credit Card (Automatic)'))
    tenure = st.number_input('Tenure')
    monthlyCharge = st.number_input('Monthly Charge')
    # making changes to match label encoded data in the dataframe
    # values are assigned as equivalent to values from an inverse_transform of the LabelEncoder
    if(gender=='Male'):
        gender = 1
    elif(gender=='Female'):
        gender=0
    if(seniorcitizen=='Yes'):
        seniorcitizen = 1
    elif(seniorcitizen=='No'):
        seniorcitizen=0
    if(internetService=='Fiber Optic'):
        internetService = 1
    elif(internetService=='DSL'):
        internetService = 2
    else:
        internetService = 0
    if(contract=='Month-to-month'):
        contract = 2
    elif(contract=='Two Year'):
        contract = 1
    else:
        contract = 0
    if(paymentMethod=='Electronic Cheque'):
        paymentMethod = 0
    elif(paymentMethod=='Mailed Cheque'):
        paymentMethod = 2
    elif(paymentMethod=='Bank Transfer (Automatic)'):
        paymentMethod = 1
    else:
        paymentMethod = 3
    # convert tenure to integer
    tenure = int(tenure)
    # store all the variables in a numpy array
    user_data = np.array([gender,seniorcitizen,internetService,contract,paymentMethod,tenure,monthlyCharge]).reshape(1,-1)
    return user_data

# training the XGBoost Classifier
@st.cache(suppress_st_warning=True)
def xgbclassifier(Xtrain,Xtest,ytrain,ytest):
    clf = XGBClassifier()
    clf.fit(Xtrain,ytrain)
    y_pred = clf.predict(Xtest)
    acc_score = metrics.accuracy_score(ytest,y_pred) * 100
    auc_score = metrics.roc_auc_score(ytest, y_pred) * 100
    # saving the model
    pickle_in = open("xgb_model.pkl", "wb")
    xgb_model = pickle.dump(clf, pickle_in)
    pickle_in.close()
    return acc_score,auc_score


def main():
    # loading the data
    data = load_data()
    Xtrain,Xtest,ytrain,ytest,churn_encoder = preprocess(data)
    accuracy, auc_score = xgbclassifier(Xtrain,Xtest,ytrain,ytest)
    # loading the presaved model
    pickle_out = open('xgb_model.pkl', "rb")
    model = pickle.load(pickle_out)
    pickle_out.close()

    st.title("Customer Churn Predictor")
    st.write(' ')
    st.write(' ')
    st.sidebar.image('laura.jpg')
    st.sidebar.title("About App")
    st.sidebar.markdown('The churn prediction model aims to assist Telecom operators in predicting customers who are most likely to churn. An Extreme Gradient Boosting (XGBoost) model is trained.')

    if(st.checkbox("Display data", False)):
        st.subheader("Showing data now...")
        st.write(f'The dataset has a shape of {data.shape}')
        st.write(data.head())
        st.write('Correlation matrix is as follows...')
        st.write(data.corr())
        

    if(st.checkbox("Display metrics summary")):
        st.subheader("Display XGB Classifier metrics...")
        st.write("Model Accuracy : ", accuracy.round(2),'%')
        st.write("Auc Score : ", auc_score.round(2),'%')

    if(st.checkbox("Want to input your own data? Tick if yes")):
        user_data = accept_user_data()
        if st.button("Make Prediction"):
            prediction = model.predict(user_data)
            pred = churn_encoder.inverse_transform(prediction)
            if(pred[0] == 0):
                pred = 'No'
            else:
                pred = 'Yes'
            st.write("Is the customer likely to churn?")
            success_string = f"Classification result : {pred}"
            st.success(success_string)
            st.write('Confidence level in result : ', accuracy.round(2), '%')


if __name__ == "__main__":
    main()

