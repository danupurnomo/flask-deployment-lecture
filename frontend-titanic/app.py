import streamlit as st
import pandas as pd
import requests

def run():
    with st.form(key='form_parameters'):
        passenger_id = st.number_input('Passenger ID', step=1)
        passenger_class = st.radio('Passenger Class', (1, 2, 3))
        passenger_name = st.text_input('Passenger Name')
        sex = st.radio('Sex', ('male', 'female'))
        age = st.number_input('Age', min_value=0, value=17)
        sibsp = st.number_input('Sibling/Spouse', min_value=0, value=0)
        parch = st.number_input('Parent/Children', min_value=0, value=0)
        ticket_number = st.text_input('Ticket Number') 
        fare = st.number_input('Fare', min_value=0, value=10) 
        cabin_number = st.text_input('Cabin Number')
        embarked = st.radio('Port of Embarkation', ('C', 'Q', 'S'))

        submitted = st.form_submit_button('Predict')
    
    # Create A New data
    data_inf = {
        'PassengerId': passenger_id,
        'Pclass': passenger_class, 
        'Name': passenger_name, 
        'Sex': sex, 
        'Age': age, 
        'SibSp': sibsp,
        'Parch': parch, 
        'Ticket': ticket_number, 
        'Fare': fare, 
        'Cabin': cabin_number, 
        'Embarked': embarked
    }

    if submitted:
        # Show Inference DataFrame
        st.dataframe(pd.DataFrame([data_inf]))
        print('[DEBUG] Data Inference : \n', data_inf)
        
        # Predict
        URL = "https://backend-titanic-danu.koyeb.app/predict"
        r = requests.post(URL, json=data_inf)

        if r.status_code == 200:
            res = r.json()
            st.write('## Prediction : ', res['label_names'])
            print('[DEBUG] Result : ', res)
            print('')
        else:
            st.write('Error with status code ', str(r.status_code))
        

if __name__ == '__main__':
    run()