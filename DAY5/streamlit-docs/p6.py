import streamlit as st

st.checkbox("yes")
st.checkbox("no")

st.button("click me")
st.radio("Select your interface:",['eth0','eth1','eth2'])


st.date_input('Travel date:')

n = st.number_input('Enter your age:',5,90)
st.write(f'n value:{n}')

