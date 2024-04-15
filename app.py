import numpy as np
import pandas as pd
import streamlit as st
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from streamlit_extras.metric_cards import style_metric_cards

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

df = pd.read_excel('example.xlsx')
X = df[['average_test_score', 'Hours_of_lessons_per_week']]
Y = df['Projects']

model = LinearRegression()

X = X.values  # conversion of X  into array
model.fit(X, Y)
predictions = model.predict(X)
# y_pred and predictions on the same data
y_pred = model.predict(X)

# Regression coefficients (Bo, B1, B2)
intercept = model.intercept_  # Bo
coefficients = model.coef_  # B1, B2

# R-squared Coefficient of determination
r2 = metrics.r2_score(Y, predictions)

# R-squared
n = len(Y)
p = X.shape[1]
adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

# SSE and SSR
sse = np.sum((Y - predictions) ** 2)
ssr = np.sum((y_pred - np.mean(Y)) ** 2)

# conclusions
with st.expander("Regression coefficient"):
    col1, col2, col3 = st.columns(3)
    col1.metric('intercept:', value=f'{intercept:.4f}', delta="(Bo)")
    col2.metric('B1 coefficient:', value=f'{coefficients[0]:.4f}', delta=" for X1 number of Dependant (B1)")
    col3.metric('B2 coefficient', value=f'{coefficients[1]:.4f}', delta=" for X2 number of Wives (B2):")
    style_metric_cards(background_color="#FFFFFF", border_left_color="#9900AD", border_color="#1f66bd",
                       box_shadow="#F71938")

with st.expander("Measure of variations"):
    col1, col2, col3 = st.columns(3)

    col1.metric('R-squared:', value=f'{r2:.4f}', delta="Coefficient of Determination")
    col2.metric('Adjusted R-squared:', value=f'{adjusted_r2:.4f}', delta="Adj[R2]")
    col3.metric('SSE:', value=f'{sse:.4f}', delta="Squared(Y-Y_pred)")
    style_metric_cards(background_color="#FFFFFF", border_left_color="#9900AD", border_color="#1f66bd",
                       box_shadow="#F71938")

with st.expander("PREDICTION TABLE"):
    result_df = pd.DataFrame(
        {
            'Name': df['Name'],
            'No of average_test_score': df['average_test_score'],
            'No of Hours_of_lessons_per_week': df['Hours_of_lessons_per_week'],
            'Done Projects | Actual Y': Y,
            'Y_predicted': predictions
        }
    )
    result_df['SSE'] = sse
    result_df['SSR'] = ssr
    st.dataframe(result_df, use_container_width=True)

# download predicted csv
df_download = result_df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="download predicted dataset",
    data=df_download,
    key="download_dataframe.csv",
    file_name="my_dataframe.csv"
)

with st.expander("residual & line of best fit"):
    # Calculate residuals
    residuals = Y - predictions
    # Create a new DataFrame to store residuals
    residuals_df = pd.DataFrame({'Actual': Y, 'Predicted': predictions, 'Residuals': residuals})
    # Print the residuals DataFrame
    st.dataframe(residuals_df, use_container_width=True)
