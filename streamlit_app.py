import streamlit as st
import pickle
import keras.models
import pandas as pd
import numpy as np

@st.cache_resource
def model_use(option):
    if option == 'Logistic Regression':
        with open('models\logistic_regreesion.pkl','rb') as file:
            model = pickle.load(file)
    elif option == 'Random Forest':
        with open('models\\random_forest.pkl','rb') as file:
            model = pickle.load(file)
    else:
        model = keras.models.load_model('models\\nn_model.h5')
    return model

with st.sidebar:
    st.title('Mental Health Diagnoses')
    option = st.selectbox("Select Model you'd like to use",('Logistic Regression','Random Forest','Neural Network'),1)
    st.markdown('---')
    st.markdown('Made by [Kathanshi Jain](https://www.linkedin.com/in/kathanshi-jain/) and [Utkarsh Sen](https://www.linkedin.com/in/utk-sen/)')


header = st.container()
features = st.container()
model_training = st.container()

with header:
    st.title('Enhancing Mental Health Diagnoses')
    
with features:
    left_col,mid_col,right_col = st.columns(3)
    
    sadness = left_col.selectbox('How often do you feel sad?',['Seldom','Sometimes','Usually','Most-often'])
    euphoric = mid_col.selectbox('How often do you feel Excited?',['Seldom','Sometimes','Usually','Most-often'])
    exhausted = right_col.selectbox('How often do you feel exhausted?',['Seldom','Sometimes','Usually','Most-often'])
    sleep_disorder = left_col.selectbox('Do you have trouble sleeping?',['Yes','No'])
    mood_swing = mid_col.selectbox('Do you have often get mood swings?',['Yes','No'])
    suicidal_thoughts = right_col.selectbox('Do you have often get suicidal thoughts?',['Yes','No'])
    anorexia = left_col.selectbox('Does stress impact your eating habits?',['Yes','No'])
    respect = mid_col.selectbox('Do you often respect the people in authority?',['Yes','No'])
    explanation = right_col.selectbox('Do you often try to explain yourself?',['Yes','No'])
    response = left_col.selectbox('Are you prone to response aggresively when in bad mood?',['Yes','No'])
    move_on = mid_col.selectbox('Are you good at moving on?',['Yes','No'])
    breakdown = right_col.selectbox('Are you prone to nervous breakdown?',['Yes','No'])
    mistakes = left_col.selectbox('Do you admit your mistakes?',['Yes','No'])
    ovethinking = mid_col.selectbox('Are you an overthinker?',['Yes','No'])
    sexual_acitvity = left_col.slider('How sexually active are you?',1,10,1)
    concentration = mid_col.slider('Rate your confidence level',1,10,1)
    optimism = right_col.slider('How optimistic are you?',1,10,1)
    df = pd.DataFrame((sadness,euphoric,exhausted,sleep_disorder,mood_swing,suicidal_thoughts,anorexia,respect,explanation,response,move_on,breakdown,mistakes,
                 ovethinking,sexual_acitvity,concentration,optimism))
with model_training:
    run = st.button('Run')
    if run:
        model = model_use(option)
        df = df.replace(({'Seldom':1,'Sometimes':2,'Usually':3,'Most-often':4}))
        df = df.replace({'Yes':1,'No':0})
        X = np.array(df).transpose()
        pred = model.predict(X)
        if pred.ndim != 1:
            pred = np.argmax(pred,axis=1)
            
        if pred == 0:
            st.write('Normal')
        if pred == 1:
            st.write('Depression')
        if pred == 2:
            st.write('Bipolar Type-2')
        if pred == 3:
            st.write('Bipolar Type-1')