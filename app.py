import numpy as np
import pickle
import streamlit as st
loaded_model = pickle.load(open(r"D:\folder\model.sav","rb"))
#creating afunction for prediction
def worker_prediction(input_data):
    
    input_data_as_numpy_array=np.asarray(input_data)
    inputdata_reshaped=input_data_as_numpy_array.reshape(1,-1)
    output=loaded_model.predict(inputdata_reshaped)
    print(output)
    if(output[0]==0):
        return 'The Worker performance is bad'
    else:
        return 'The worker performance is good'
    
def main():
    st.title("Construction Worker Performance Predictor")
    Age                               = st.text_input("Enter worker age ")   
    Activity_Category                 = st.text_input("Enter workers activity category")  
    StdWorkingHrsPerDay               = st.text_input("Enter company standard working hours in a day ") 
    Pulse_rate_in_idle                = st.text_input("Enter worker pulse rate when idle ")  
    Pulse_rate_in_activity            = st.text_input("Enter worker pulse rate when in activity ")  
    Calories_Burnt_per_hour           = st.text_input("Calories burnt per hour ")  
    TargetedWorkPerWorkingDay         = st.text_input("Targeted Work Per Day out of 10")  
    Actual_Work_Done_Without_a_FITBIT = st.text_input("Actual workdone before introducing fitbit")  
    Actual_Workdone_when_using_fitbit = st.text_input("Workdone by Worker after using fitbit ")
    #code for prediction
    p= ""
    if st.button("Result"):
        p=worker_prediction([Age,Activity_Category,StdWorkingHrsPerDay,Pulse_rate_in_idle,Pulse_rate_in_activity,Calories_Burnt_per_hour,TargetedWorkPerWorkingDay,Actual_Work_Done_Without_a_FITBIT,Actual_Workdone_when_using_fitbit])
    st.success(p)
    
if __name__ == '__main__':
    main()
    