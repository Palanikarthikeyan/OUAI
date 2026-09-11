import streamlit as st


vector = st.slider('select your vector range:',500,900)
if vector:
    st.write(f'selected vector range:{vector}')

st.slider("select your age:")
st.slider("select your port:",5000,6000,5560)


st.select_slider("Select your rate:",["Bad","Good","Excellent"])

st.selectbox("select your model:",["gpt4.0","gpt5.0","lamma","gemma2:2b"])
st.multiselect("select your model:",["gpt4.0","gpt5.0","lamma","gemma2:2b"])
