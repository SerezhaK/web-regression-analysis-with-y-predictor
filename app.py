import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


st.set_option('deprecation.showPyplotGlobalUse', False)
from streamlit_extras.metric_cards import style_metric_cards

# navicon and header
st.set_page_config(page_title="Dashboard", page_icon="📈", layout="wide")

st.header("Machine learning workflow")
st.write("Multiple regression with SSE, SE, SSR, SST, R2, ADJ[R2], residual")

with open('styles.css') as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
