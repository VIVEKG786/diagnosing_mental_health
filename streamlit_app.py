# Importing libraries
import streamlit as st
import pickle
import time
import keras.models
import pandas as pd
import numpy as np


# Function to load model
@st.cache_resource
def model_use(option):
    if option == 'Logistic Regression':
        with open('models/logistic_regreesion.pkl','rb') as file:
            model = pickle.load(file)
    elif option == 'Random Forest':
        with open('models/random_forest.pkl','rb') as file:
            model = pickle.load(file)
    else:
        model = keras.models.load_model('models/nn_model.h5')
    return model

# Sidebar
with st.sidebar:
    st.title('Mental Health Diagnoses')
    st.markdown('---')
    option = st.radio("Select Model you'd like to use",('Logistic Regression','Random Forest','Multi Layer Perceptron'),1)
    st.markdown('---')
    st.markdown('Made by [Kathanshi Jain](https://www.linkedin.com/in/kathanshi-jain/) and [Utkarsh Sen](https://www.linkedin.com/in/utk-sen/)')

# Title of the project
st.title('Mental Health Diagnoses')
st.image('images/front.jpg')

# Tabs
about,diagnosis, faq= st.tabs(['About','Diagnosis','FAQs'])

# About Tab
with about:
    st.title('About Project')
    
    st.markdown('''The dataset comprised 30 samples for each of the **Normal**, **Mania Bipolar Disorder**, **Depressive Bipolar Disorder**, and **Major Depressive Disorder** 
             categories summing up to **120 patiants**. The dataset contains the 17 essential symptoms psychiatrists use to diagnose the described disorders. 
             The behavioral *symptoms considered* are the levels of patients *Sadness*, *Exhaustness*, *Euphoric*, *Sleep disorder*, *Mood swings*, *Suicidal thoughts*, *Anorexia*, 
             *Anxiety*, *Try-explaining*, *Nervous breakdown*, *Ignore & Move-on*, *Admitting mistakes*, *Overthinking*, *Aggressive response*, *Optimism*, *Sexual activity*, and *Concentration* in a 
             Comma Separated Value (CSV) format. The Normal category refer to the individuals using therapy time for specialized 
             counseling, personal development, and life skill enrichments. While such individuals may also have minor mental problems, they differ 
             from those suffering from Major Depressive Disorder and Bipolar Disorder.''')
    st.markdown('''***Note:*** This project is only for the learning pursposes. The dataset used in the training is from secondary source. 
                The decision made by the model is not final. Please [consult profesionals](https://www.thelivelovelaughfoundation.org/find-help/helplines)
                for accurate diagnoses.''')
    
    st.subheader('Problem Statement')
    st.markdown('Classify patients into ***Normal***, ***Depressed***, ***Bipolar Type-1***, and ***Bipolar Type-2*** based on the above mentioned features.')
    
    st.subheader('Models Used')
    st.markdown('Models used for the classification are:')
    with st.expander('Logistic Regression'):
        # Logistic Regression Explanation
        st.image('images/sigmoid_function.png')
        
        st.markdown('''Logistic regression is a statistical method that uses math to find the relationship between two data factors and predict the value of one 
                    factor based on the other. The model is similar to linear regression but instead of returning continous values, it returns descrete anf finite 
                    values, like yes or no.''')
        st.markdown('''Logistic Regression uses sigmoid function to map the relationship between dependent and independent variables. Sigmoid transforms real numbers into
                    probabilities between 0 and 1.''')
        # Sigmoid Function
        st.markdown('''$$f(z) =\\frac{1}{1+e^{-z}}$$''')
        # Accuracy
        st.markdown('**Accuracy**')
        st.markdown('Logistic regression performed with *85% accuracy*.')
        # Parameters
        st.markdown('**Parameter**')
        st.markdown('***C(Inverse of learning rate):*** 20.69')
        st.markdown('***Rgularizer:*** L2')
        st.markdown('***Solver:*** liblinear')
        
    with st.expander('Random Forest Classifier'):
        # Random Forest
        st.image('images/random_forest.png')
        
        st.markdown('''A random forest (RF) classifier is a machine learning ensemble learning method that uses multiple decision trees to classify data. Each tree is created 
                    from a random vector sampled from the input vector, and each tree votes for the most popular class to classify the input vector. The tree with the highest 
                    probability is selected as the output of the random forest.''')
        # Accuracy
        st.markdown('**Accuracy**')
        st.markdown('Random Forest performed with *93% accuracy*.')
        # Parameters
        st.markdown('**Parameter**')
        st.markdown('***Max Depth:*** 4')
        st.markdown('***Minimum Sample Leaf:*** 2')
        
    with st.expander('Multi Layer Perceptron'):
        # MLP
        st.image('images/mlp_neuralnetwork.png')
        
        st.markdown('''A multilayer perceptron (MLP) is a type of artificial neural network (ANN) that consists of at least three layers of fully connected neurons with 
                    nonlinear activation functions. MLPs are known for their ability to distinguish data that is not linearly separable''')
        # Accuracy
        st.markdown('**Accuracy**')
        st.markdown('Multilayer Perceptron performed with *87% accuracy*.')
        # Parameters
        st.markdown('**Parameter**')
        st.markdown('***Number of hidden layers:*** 3')
        st.markdown('***Function used in hidden layers:*** tanh')
        st.markdown('***Drop out rate:*** 10%')
        st.markdown('***Output Function:*** Softmax')
    
    # How to use    
    st.subheader('How to use')
    st.write('**Step 1:** Select one of the following model from the side bar:')
    df = pd.DataFrame({'Model':['Logistic Regression','Random Forest Classifier','Multi Layer Perceptron'],
                       'Accuracy':['85%','93%','87%']})
    st.table(df)
    st.write('**Step 2:** Go to diagnosis tab.')
    st.write('**Step 3:** Fill all the fields')
    st.write('**Step 4:** Hit the run button to generate diagnosis.')
    st.write('**Step 5:** Drop down Know more section to find out more. Check professional help section for better analysis.')

# Diagnosis
with diagnosis:
    header = st.container()
    features = st.container()
    model_training = st.container()
    with header:
        st.title('Diagnose Center')
    with features:
        left_col,mid_col,right_col = st.columns(3)
        
        sadness = left_col.selectbox('How often do you feel sad?',['Seldom','Sometimes','Usually','Most-often'])
        euphoric = mid_col.selectbox('How often do you feel Excited?',['Seldom','Sometimes','Usually','Most-often'])
        exhausted = right_col.selectbox('How often do you feel exhausted?',['Seldom','Sometimes','Usually','Most-often'])
        sleep_disorder = left_col.selectbox('How often do you have trouble sleeping?',['Seldom','Sometimes','Usually','Most-often'])
        mood_swing = mid_col.selectbox('Do you often get mood swings?',['Yes','No'])
        suicidal_thoughts = right_col.selectbox('Do you have often get suicidal thoughts?',['Yes','No'])
        anorexia = left_col.selectbox('Does stress impact your eating habits?',['Yes','No'])
        respect = mid_col.selectbox('Do you often respect the people in authority?',['Yes','No'])
        explanation = right_col.selectbox('Do you often try to explain yourself?',['Yes','No'])
        response = left_col.selectbox('Are you prone to response aggresively when in bad mood?',['Yes','No'])
        move_on = mid_col.selectbox('Are you good at ignoring problems and moving on?',['Yes','No'])
        breakdown = right_col.selectbox('Are you prone to nervous breakdown?',['Yes','No'])
        mistakes = left_col.selectbox('Do you admit your mistakes?',['Yes','No'])
        ovethinking = mid_col.selectbox('Are you an overthinker?',['Yes','No'])
        sexual_acitvity = left_col.slider('How sexually active are you?',1,10,1)
        concentration = mid_col.slider('Rate your confidence level',1,10,1)
        optimism = right_col.slider('How optimistic are you?',1,10,1)
        
        # Creating DataFrame of the fields for ease of cleaning
        df = pd.DataFrame((sadness,euphoric,exhausted,sleep_disorder,mood_swing,suicidal_thoughts,anorexia,respect,explanation,response,move_on,breakdown,mistakes,
                    ovethinking,sexual_acitvity,concentration,optimism))
        
    with model_training:
        run = st.button('Run')
        if run:
            latest_iteration = st.empty()
            bar = st.progress(0)      # For progress bar
            for i in range(100):
                model = model_use(option)
                df = df.replace(({'Seldom':1,'Sometimes':2,'Usually':3,'Most-often':4}))
                df = df.replace({'Yes':1,'No':0})
                X = np.array(df).transpose()
                pred = model.predict(X)
                if pred.ndim != 1:
                    pred = np.argmax(pred,axis=1)
                latest_iteration.text(f'{i+1}%')
                bar.progress(i+1)
                time.sleep(0.01)
            
            st.divider()
            # Result    
            if pred == 0:
                st.success("**You are diagnosed as Normal**")
            if pred == 1:
                st.warning("**You are diagnosed as Depressed**")
            if pred == 2:
                st.warning('**You are diagnosed as Bipolar Type-2**')
            if pred == 3:
                st.error("**You are diagnosed as Bipolar Type-1.**")
            
            with st.expander('Know More'):
                if pred == 0:
                    # Diagnosed Normal
                    st.markdown("""The Normal category refer to the individuals using therapy time for specialized 
                                counseling, personal development, and life skill enrichments. While such individuals 
                                may also have minor mental problems, they differ from those suffering from Major Depressive 
                                Disorder and Bipolar Disorder.""")
                    st.markdown("[know more](https://en.wikipedia.org/wiki/Normality_(behavior)#:~:text=Normality%20is%20a%20behavior%20that,society%20(known%20as%20conformity).)")
                    st.markdown('**Seek professional help:**')
                    st.markdown("- *[Manastha](https://www.manastha.com/)*")
                    st.markdown("- *[Vandrevala Foundation(Free)](https://www.vandrevalafoundation.com/free-counseling)*")
                    st.markdown("- *[Live Love Laugh Foundation](https://www.thelivelovelaughfoundation.org/find-help/helplines)*")
                if pred == 1:
                    # Diagnosed Depressed
                    st.markdown("""Depression is a mental health disorder characterized by persistent feelings of sadness, hopelessness, 
                                and a loss of interest or pleasure in activities. It is more than just feeling sad or going through a rough 
                                patch—it is a serious medical condition that affects how a person feels, thinks, and behaves, and it can interfere 
                                with daily functioning.""")
                    st.markdown("""Depression can vary in severity and may be episodic or chronic. It can also co-occur with other mental health 
                                disorders, such as anxiety disorders or substance use disorders. Depression can have a profound impact on all aspects 
                                of a person's life, including relationships, work or school performance, and physical health.""")
                    st.markdown('**Seek professional help:**')
                    st.markdown("- *[Manastha](https://www.manastha.com/)*")
                    st.markdown("- *[Vandrevala Foundation(Free)](https://www.vandrevalafoundation.com/free-counseling)*")
                    st.markdown("- *[Live Love Laugh Foundation](https://www.thelivelovelaughfoundation.org/find-help/helplines)*")
                if pred == 2:
                    # Diagnosed Bipolar Type-2
                    st.markdown("""Bipolar disorder type 2 is a mood disorder characterized by recurrent episodes of depression and hypomania.
                                Hypomania is a mood state characterized by a distinct period of elevated, expansive, or irritable mood and increased 
                                energy or activity levels. While hypomania is less severe than mania, it still represents a significant deviation from 
                                an individual's baseline mood and functioning. """)
                    st.markdown("""Individuals with bipolar disorder type 2 experience cycling between depressive and hypomanic episodes, which can 
                                significantly impair functioning. Treatment typically involves a combination of medication and psychotherapy.""")
                    st.markdown('**Seek professional help:**')
                    st.markdown("- *[Manastha](https://www.manastha.com/)*")
                    st.markdown("- *[Vandrevala Foundation(Free)](https://www.vandrevalafoundation.com/free-counseling)*")
                    st.markdown("- *[Live Love Laugh Foundation](https://www.thelivelovelaughfoundation.org/find-help/helplines)*")
                if pred == 3:
                    # Diagnosed Bipolar Type-1
                    st.markdown("""Bipolar disorder type 1, often referred to simply as bipolar I disorder, is a mental health condition characterized 
                                by mood swings that include episodes of mania and depression.A person affected by bipolar I disorder has had at least 
                                one manic episode in their life. A manic episode is a period of abnormally elevated or irritable mood and high energy, 
                                accompanied by abnormal behavior that disrupts life.Most people with bipolar I disorder also suffer from episodes of 
                                depression. Often, there is a pattern of cycling between mania and depression. This is where the term "manic depression" 
                                comes from. In between episodes of mania and depression, many people with bipolar I disorder can live normal lives.""")
                    st.markdown('**Seek professional help:**')
                    st.markdown("- *[Manastha](https://www.manastha.com/)*")
                    st.markdown("- *[Vandrevala Foundation(Free)](https://www.vandrevalafoundation.com/free-counseling)*")
                    st.markdown("- *[Live Love Laugh Foundation](https://www.thelivelovelaughfoundation.org/find-help/helplines)*")

# FAQs                        
with faq:
    st.title('Frequently Asked Questions')
    with st.expander('**What is the source of the dataset?**'):
        st.markdown('''The dataset is used from the Harvard Dataverse. [Click link](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/0FNET5)
                    to access''')
        
    with st.expander('**Whose data was collected?**'):
        st.markdown('''The current data was collected from a private psychology clinic by the people of Harvard. 
                    The dataset comprised 30 samples belonging to each category summing upto 120 patients.''')
        
    with st.expander('**Does it provide accurate diagnosis?**'):
        st.markdown('''There is nothing like completely accurate prediction in machine learning. So, no. Although all the models have accuracy above 80%, 
                    the diagnosis should not be considered completely trustworthy. To get a better diagnosis, [click here](https://www.thelivelovelaughfoundation.org/find-help/helplines)''')
        
    with st.expander('**Can we start medication based on the diagnosis?**'):
        st.markdown('''This project is only for **learning purposes**. It does not claim trustworthiness of the result.
                    Please consult a professional before starting medication.''')
        
    with st.expander('**Why only 3 models were used?**'):
        st.markdown('''5 models were trained on the dataset: *Logistic Regression*, *Support Vector Classifier*,
                    *K-Nearest Neighborhood*, *Random Forest Classifier*, and *Multilayer Perceptron*. Of all the 5 models, 
                    *Logistic Regression*,*Random Forest Classifier*, and *Multilayer Perceptron* had the best accuracy and recall.
                Therefore, only these 3 models were used for deployement.''')
        
    with st.expander('**Where can I get connected with the authors?**'):
        st.markdown('''You can reach out to us at [Kathanshi Jain](https://www.linkedin.com/in/kathanshi-jain/)
                    and [Utkarsh Sen](https://www.linkedin.com/in/utk-sen/)''')