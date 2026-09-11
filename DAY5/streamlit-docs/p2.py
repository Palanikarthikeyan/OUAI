import streamlit as st
import pandas as pd

product={}
product['pid']=[101,112,302,120]
product['pname']=['pA','pB','pC','pD']
product['pcost']=[1000,500,2300,3120]
df = pd.DataFrame(product)

st.write('product details:-')
st.write(df)

import numpy as np
data = pd.DataFrame(np.random.randn(20,5),columns=['F1','F2','F3','F4','F5'])
st.write(data)

st.write("## Top 5 records:",data.head())

