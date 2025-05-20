import streamlit as st
import pandas as pd
import pickle

import subprocess
import sys

# Full path to your requirements.txt file
requirements_path = r"C:\Users\Saurabh Rai\Desktop\Work\IPL-match-Winning-Prediction--main\Deployment\requirements.txt"

try:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_path])
    print("All packages installed successfully.")
except subprocess.CalledProcessError as e:
    print(f"Error occurred while installing packages: {e}")

# Declaring the teams
teams = ['Royal Challengers Bangalore', 'Mumbai Indians', 'Kings XI Punjab', 'Kolkata Knight Riders',
       'Sunrisers Hyderabad', 'Rajasthan Royals', 'Chennai Super Kings',
       'Delhi Capitals']

# decarling the venues

cities = ['Chandigarh', 'Chennai', 'Bangalore', 'Mumbai', 'Dharamsala',
       'Hyderabad', 'Cuttack', 'Jaipur', 'Raipur', 'Delhi', 'Nagpur',
       'Kolkata', 'Indore', 'Centurion', 'Ahmedabad', 'Abu Dhabi',
       'East London', 'Durban', 'Pune', 'Visakhapatnam', 'Mohali',
       'Johannesburg', 'Cape Town', 'Bengaluru', 'Sharjah',
       'Port Elizabeth', 'Kimberley', 'Ranchi', 'Bloemfontein']

pipe = pickle.load(open(r'C:\Users\Saurabh Rai\Desktop\Work\IPL-match-Winning-Prediction--main\Deployment\pipe.pkl', 'rb'))

st.title("IPL Win Predictor")

col1, col2 = st.columns(2)

with col1:
    battingteam = st.selectbox('select the batting team', sorted(teams))

with col2:
    bowlingteam = st.selectbox('Select the bowling team', sorted(teams))

city = st.selectbox('select the city where the match is being played', sorted(cities))

target = st.number_input('Target') 

col3, col4, col5 = st.columns(3)

with col3:
    score = st.number_input('Score')

with col4:
    overs = st.number_input('Overs Completed')

with col5:
    wickets = st.number_input('Wickets Fallen')

if st.button("Predict Probability"):
    runs_left = target - score
    balls_left = 120 - (overs*6)
    wickets = 10 - wickets
    currentrunrate = score/overs
    requiredrunrate = (runs_left*6)/balls_left

    input_df = pd.DataFrame({'batting_team':[battingteam], 'bowling_team':[bowlingteam],
                             'city':[city], 'runs_left':[runs_left],
                             'balls_left':[balls_left],'wickets':[wickets],
                             'total_runs_x':[target],'current_run_rate':[currentrunrate],
                             'req_run_rate':[requiredrunrate]})
    result = pipe.predict_proba(input_df)
    lossprob = result[0][0]
    winprob = result[0][1]

    st.header(battingteam+" - "+str(round(winprob*100))+"%")
    st.header(bowlingteam+" - "+str(round(lossprob*100))+"%")
