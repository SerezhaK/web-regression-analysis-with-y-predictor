import streamlit as st

st.set_option('deprecation.showPyplotGlobalUse', False)

# navicon and header
st.set_page_config(page_title="Dashboard", page_icon="📈", layout="wide")

st.header("Machine learning workflow")
st.write("Multiple regression with SSE, SE, SSR, SST, R2, ADJ[R2], residual")

with open('styles.css') as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# logo
st.sidebar.image("images/logo_1.webp", caption="multi—variable regression")
st.sidebar.title("add new variable")
