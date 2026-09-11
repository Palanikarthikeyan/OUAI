import streamlit as st
import numpy as np
import pandas as pd

data = pd.DataFrame(np.random.randn(20,5),columns=['F1','F2','F3','F4','F5'])
st.write(data)

st.write("## Top 3 records:",data.head(3))

# display chart

st.line_chart(data)

