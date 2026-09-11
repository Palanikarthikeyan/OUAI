import streamlit as st

st.title('Welcome to streamlit app')
st.write("This is test message1")
st.write("This is test message2")

model = st.text_input('Enter your model name:')
#st.write(f"Input model name is:{model}")

if model:
    st.write(f"input model name is:{model}")

name = st.text_input('Enter your name:')

if name:
    st.success(f'Hello..{name} how are you')


st.markdown("*** Bold Text ***")

if st.button("Click Me"):
	st.success("Task Done")
	st.balloons()


### Text and Display 
#---------------------------------------