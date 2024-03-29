import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import numpy as np
import seaborn as sns
import io

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split



st.set_page_config(page_title = "Name of Service Line",
                   page_icon = ':bar_chart:',
                   layout = "wide")


#----------Supporting Functions---------

def download_image(img_key):
    b = io.BytesIO()
    plt.savefig(b, format='png')
    plt.close()

    btn = st.download_button(
             label="Download Chart",
             data=b.getvalue(),
             file_name="Chart_Image.png",
             mime="image/png",
             key=img_key)

    return btn



# ---- MAINPAGE -----

st.title(":bar_chart: Service Line Dashboard")
st.markdown("##")

st.subheader("Any Caption or Date updated etc")
st.markdown("---")

data_dcode = pd.read_csv('dcode.csv', parse_dates=True).reset_index(drop=True)
#data_dcode.index = pd.to_datetime(data_dcode.index)


data_credit = pd.read_csv('credit.csv', parse_dates=True)
# data_credit['Date'] = pd.to_datetime(data_credit["Date"].dt.strftime('%Y-%m-%d'))

# --- sidebar---


st.sidebar.header("Please select filter")

dcode = st.sidebar.multiselect(
    "Select d-code :",
    options = data_dcode['DCode'].unique(),
    default = data_dcode['DCode'].unique(),
    key="1")



credit = st.sidebar.multiselect(
    "Select Credit Location :",
    options = data_credit['Credit'].unique(),
    default = data_credit['Credit'].unique(),
    key="2")


d_code_selection = data_dcode.query("DCode == @dcode")
d_code_selection['Date'] = pd.to_datetime(d_code_selection['Date']).dt.strftime('%Y/%m/%d') 

credit_selection = data_credit.query("Credit == @credit")
credit_selection['Date'] = pd.to_datetime(credit_selection['Date']).dt.strftime('%Y/%m/%d')

try:
    start_date, end_date = st.select_slider(
         'Select a range of date',
         options= d_code_selection['Date'].sort_values(ascending=True),
         value=(d_code_selection['Date'].iloc[-1], d_code_selection['Date'][0]))
    st.write('You have selected range between', start_date, 'and', end_date)

except:
    "Please Select a D-Code"


try:
    d_code_selection = d_code_selection.query('Date >= @start_date and Date <= @end_date')
    credit_selection = credit_selection.query('Date >= @start_date and Date <= @end_date')

except:
    "Please select a Dcode"





# ---- Final dataframe ----
date_form = DateFormatter("%m-%Y")
left_column, right_column = st.columns(2)
    

# ---------DCode side ---------

with left_column:
    st.subheader('dcode_table')
    st.dataframe(d_code_selection)

    download_file = d_code_selection.to_csv().encode('utf-8')
    st.download_button(
         label="Download DCode as CSV",
         data = download_file,
         file_name='selected_dcode.csv')
    
    # ---- Charts ------
    st.markdown("---")
    st.subheader('D-Code Charts')
   
    d_code_selection[['Average Price-2020', 'Average Price-2021', 'Average Price-2022', 'Closing Value-2020','Closing Value-2021','Closing Value-2022']] = d_code_selection[['Average Price-2020', 'Average Price-2021', 'Average Price-2022', 'Closing Value-2020','Closing Value-2021','Closing Value-2022']].apply(lambda x: x.str.replace('$', '')).astype(float)
    d_code_selection['Date'] = pd.to_datetime(d_code_selection['Date'])
    
    
    # Average Price-2020
    fig_dcode_ap20, ax_dcode = plt.subplots()
    sns.lineplot(x="Date", y='Average Price-2020', hue="DCode",  data=d_code_selection,  ax=ax_dcode) #, alpha=.5, ax=ax2,
    fig_dcode_ap20.tight_layout()
    plt.xticks(rotation=60)
    ax_dcode.xaxis.set_major_formatter(date_form)
    st.pyplot(fig_dcode_ap20)

    download_image("ap20")
    
    # Average Price-2021    
    fig_dcode_ap21, ax_dcode = plt.subplots()
    sns.lineplot(x="Date", y='Average Price-2021', hue="DCode",  data=d_code_selection)
    fig_dcode_ap21.tight_layout()
    plt.xticks(rotation=60)
    ax_dcode.xaxis.set_major_formatter(date_form)
    st.pyplot(fig_dcode_ap21)
    
    download_image("ap21")
    
    # Average Price-2022    
    fig_dcode_ap22, ax_dcode = plt.subplots()
    sns.lineplot(x="Date", y='Average Price-2022', hue="DCode", data=d_code_selection)#, palette='Reds')    
    fig_dcode_ap22.tight_layout()
    plt.xticks(rotation=60)
    ax_dcode.xaxis.set_major_formatter(date_form)
    st.pyplot(fig_dcode_ap22)

    download_image("ap22")


st.markdown("---")
   
    
# ---------- Credit Side ----------

with right_column:
    st.subheader('credit_table')
    st.dataframe(credit_selection)

    download_file = credit_selection.to_csv().encode('utf-8')
    st.download_button(
         label="Download Credit data as CSV",
         data= download_file,
         file_name='selected_credit.csv')

    # ---- Charts ------
    st.markdown("---")
    st.subheader('Credit Charts')


    credit_selection[['Average Price', 'Closing Value']] = credit_selection[['Average Price', 'Closing Value']].apply(lambda x: x.str.replace('$', '')).astype(float)
    credit_selection['Date'] = pd.to_datetime(credit_selection['Date'])
    fig, axes = plt.subplots() #figsize=(15, 5)

    sns.lineplot(ax=axes, x="Date", y='Average Price', hue="Credit",  data=credit_selection)
    plt.xticks(rotation=60)
    axes.xaxis.set_major_formatter(date_form)
    st.pyplot(fig)

    download_image("apcr")
    





# ---- Extra required --- 
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


