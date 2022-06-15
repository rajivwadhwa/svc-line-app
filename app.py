import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import io
from matplotlib.dates import DateFormatter ,DayLocator
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split



st.set_page_config(page_title = "RIN and LCFS Environmental Attribute Markets Dashboard - Renewable Natural Gas Revenue Driver",
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

st.title(":bar_chart: RIN and LCFS Environmental Attribute Markets Dashboard - Renewable Natural Gas Revenue Driver")
st.markdown("##")

st.subheader("*Investment, Policy & Economics Service Line, Americas.* \n**Key Contacts: Nikhil Khurana & Kandasamy Sivasubramanian**")
st.markdown(""" **Note**

             This dashboard contains up to date information on market pricing for :

                1. Renewable Identification Number (RIN) credits regulated under the Federal Renewable Fuel Standard (RFS).
                2. Low Carbon Fuel Standard (LCFS) credits regulated under the California LCFS program.
                3. Clean Fuels Program (CFP) credits regulated under Oregons CFP program.

    In simple terms, these credits, also called environmental attributes, are generated when Renewable Natural Gas (RNG) is produced and sold to transportation users.
    These credits typically provide > 80% of the revenue for an RNG project, and are key drivers of project feasibility. 

    *Source of information: EcoEngineers Daily RIN, LCFS & CFP Update Reports. This source should be credited if this information is shared externally.
    """)

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
days = DayLocator(interval=14)
left_column, right_column = st.columns(2)
    

# ---------DCode side ---------
   
d_code_selection[['Average Price-2020', 'Average Price-2021', 'Average Price-2022', 'Closing Value-2020','Closing Value-2021','Closing Value-2022']] = d_code_selection[['Average Price-2020', 'Average Price-2021', 'Average Price-2022', 'Closing Value-2020','Closing Value-2021','Closing Value-2022']].apply(lambda x: x.str.replace('$', '')).astype(float)
d_code_selection['Date'] = pd.to_datetime(d_code_selection['Date'])

avg_price_mean_column = d_code_selection.loc[: , 'Average Price-2020':'Average Price-2022']
closing_value_mean_column = d_code_selection.loc[: , 'Closing Value-2020':'Closing Value-2022']
d_code_selection['Average Price'] = avg_price_mean_column.mean(axis=1).round(2)
d_code_selection['Closing Value'] = avg_price_mean_column.mean(axis=1).round(2)


to_show_dcode = d_code_selection[['DCode','Date','Average Price','Closing Value']]
to_show_dcode[['Average Price','Closing Value']] = to_show_dcode[['Average Price','Closing Value']].apply(lambda x: '$'+ x.astype(str))
to_show_dcode['Date'] = pd.to_datetime(to_show_dcode['Date']).dt.strftime('%Y/%m/%d')

with left_column:
    st.subheader('RIN Price Data')
    st.dataframe(to_show_dcode)

    download_file = d_code_selection.to_csv().encode('utf-8')
    st.download_button(
         label="Download DCode as CSV",
         data = download_file,
         file_name='selected_dcode.csv')
    
    # ---- Charts ------
    st.markdown("---")
    st.subheader('RIN Price Charts')
    
    # Average Price
    fig_dcode_ap20, ax_dcode = plt.subplots()
    sns.lineplot(x="Date", y='Average Price', hue="DCode",  data=d_code_selection,  ax=ax_dcode ) #, alpha=.5, ax=ax2, ,  palette='husl'
    plt.xticks(rotation=60)
    ax_dcode.xaxis.set_major_formatter(date_form)
    plt.xlabel(" ")
    plt.ylabel("RIN Price ($/credit)")
    
    st.pyplot(fig_dcode_ap20)

    download_image("ap20")

st.markdown("---")
   
    
# ---------- Credit Side ----------

with right_column:
    st.subheader('California LCFS and Oregon CFP Price Data')
    st.dataframe(credit_selection)

    download_file = credit_selection.to_csv().encode('utf-8')
    st.download_button(
         label="Download Credit data as CSV",
         data= download_file,
         file_name='selected_credit.csv')

    # ---- Charts ------
    st.markdown("---")
    st.subheader('California LCFS and Oregon CFP Price Charts')


    credit_selection[['Average Price', 'Closing Value']] = credit_selection[['Average Price', 'Closing Value']].apply(lambda x: x.str.replace('$', '')).astype(float)
    credit_selection['Date'] = pd.to_datetime(credit_selection['Date'])
    fig, axes = plt.subplots() #figsize=(15, 5)

    sns.lineplot(ax=axes, x="Date", y='Average Price', hue="Credit",  data=credit_selection, palette='BuPu')
    plt.xticks(rotation=60)
    axes.xaxis.set_major_formatter(date_form)
    plt.xlabel(" ")
    plt.ylabel("Credit prices ($/MTCO2eq)")
    st.pyplot(fig)

    download_image("apcr")
    

# -------------Extra -------------

st.subheader("Additional notes and references")
st.markdown("---")




# ---- Extra required --- 
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


