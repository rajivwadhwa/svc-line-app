import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import io
import matplotlib.ticker as ticker
from matplotlib.dates import DateFormatter ,DayLocator, MonthLocator
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split



st.set_page_config(page_title = "RIN and LCFS Environmental Attribute Markets Dashboard - Renewable Natural Gas Revenue Drivers",
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

st.title(":bar_chart: RIN and LCFS Environmental Attribute Markets Dashboard - Renewable Natural Gas Revenue Drivers")
st.markdown("##")

st.subheader("*Investment, Policy & Economics Service Line, Americas. Key Contacts: Nikhil Khurana & Kandasamy Sivasubramanian*")
st.markdown(""" This dashboard contains up to date information on market pricing for :  
                1.Renewable Identification Number (RIN) credits regulated under the Federal Renewable Fuel Standard (RFS).  
                2.Low Carbon Fuel Standard (LCFS) credits regulated under the California LCFS program.  
                3.Clean Fuels Program (CFP) credits regulated under Oregons CFP program.  
    In simple terms, these credits, also called environmental attributes, are generated when Renewable Natural Gas (RNG) is produced and sold to transportation users.
    These credits typically provide > 80% of the revenue for an RNG project, and are key drivers of project feasibility. 

    *Source of information: EcoEngineers Daily RIN, LCFS & CFP Update Reports. This source should be credited if this information is shared externally.
    """)

st.markdown(""" **Notes on RIN Price information :**

    -- There are several different types of RINs, differentiated by their 'd-code'. The RIN types relevant for RNG projects are D5 RINs (where food waste feedstock is used) and D3 RINs (landfill gas, manure, wastewater feedstocks).  
    -- RINs have an associated 'vintage', which relates to the year in which the RIN credit is 'retired' by its holder to meet its compliance needs. Vintage is not important when assessing future RNG project feasibility. The prices shown for RIN credits below are the average values across the three vintage years for which data is available.
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

# ----Date Slider------
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

#----- Update Remark ----
st.markdown(f"""Date of last update = {data_dcode['Date'].iat[0]} .  Price information is up to this date.
If you require a data refresh to today’s date, please contact Kandasamy Sivasubramanian.""")

# ---- Final dataframe ----
date_form = DateFormatter("%m-%d-%Y")
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
    st.subheader('RIN Price Chart')
    
    # Average Price

    fig_dcode_ap20, ax_dcode = plt.subplots()
    sns.lineplot(x="Date", y='Average Price', hue="DCode",  data=d_code_selection,  ax=ax_dcode ) #, alpha=.5, ax=ax2, ,  palette='husl'

    # Graph Settings
    ax_dcode.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax_dcode.yaxis.set_minor_locator(ticker.AutoMinorLocator(0.1))
    ax_dcode.yaxis.set_major_formatter('${x:1.2f}')
    
    ax_dcode.xaxis.set_minor_locator(DayLocator(interval=7))
    ax_dcode.xaxis.set_major_locator(MonthLocator(interval=1))
    ax_dcode.xaxis.set_major_formatter(date_form)
    
    ax_dcode.grid(which='minor', alpha=0.2)
    ax_dcode.grid(which='major', alpha=0.5)
    
    plt.xticks(rotation=60)    
    plt.xlabel(" ")
    plt.ylabel("RIN Price ($/credit)")
    plt.legend(loc='upper left')
    
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
    st.subheader('California LCFS, Oregon CFP Price Chart')


    credit_selection[['Average Price', 'Closing Value']] = credit_selection[['Average Price', 'Closing Value']].apply(lambda x: x.str.replace('$', '')).astype(float)
    credit_selection['Date'] = pd.to_datetime(credit_selection['Date'])
    fig, axes = plt.subplots() #figsize=(15, 5)

    sns.lineplot(ax=axes, x="Date", y='Average Price', hue="Credit",  data=credit_selection, palette='BuPu')

    # Graph Settings
    axes.yaxis.set_major_locator(ticker.MultipleLocator(20))
    axes.yaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    axes.yaxis.set_major_formatter('${x:1.2f}')
    
    axes.xaxis.set_minor_locator(DayLocator(interval=7))
    axes.xaxis.set_major_locator(MonthLocator(interval=1))
    axes.xaxis.set_major_formatter(date_form)
    
    axes.grid(which='minor', alpha=0.2)
    axes.grid(which='major', alpha=0.5)

    plt.xticks(rotation=60)
    plt.xlabel(" ")
    plt.ylabel("Credit prices ($/MTCO2eq)")

    st.pyplot(fig)

    download_image("apcr")
    

# -------------Extra -------------

st.subheader("Additional notes and references")
st.markdown("---")

st.markdown("""
Additional references:

 - Introductory information on RIN credits and the RFS program: https://www.sourcena.com/sourceline/the-abcs-of-the-rfs-and-rins/
 - Introductory information on the California LCFS credits and program: https://ww2.arb.ca.gov/resources/documents/lcfs-basics
 - Introductory information on the Oregon CFP credits and program: https://www.turnermason.com/newsletter/snapshot-lcfs-california-versus-oregon/
 - Historical price points for RIN credits (pre-2021): https://www.epa.gov/fuels-registration-reporting-and-compliance-help/rin-trades-and-price-information
 - Historical price points for California LCFS and Oregon CFP credits (pre-2021): https://www.turnermason.com/newsletter/snapshot-lcfs-california-versus-oregon/

Please liaise with our Commercial Team within GHD Advisory and our Future Energy teams to get in contact with our inhouse specialists who have deep familiarity with these programs.
""")
st.markdown("---")

# ---Predicting DCode df----
from datetime import date, timedelta
st.subheader("Data Intelligence")

d_code_selection['Date'] = pd.to_datetime(d_code_selection['Date'])  
d_code_selection['Date_Delta'] = (d_code_selection['Date'] - d_code_selection['Date'].min())  / np.timedelta64(1,'D')
date_predict = date.today() + timedelta(days=1)


def model(df, X_test):
    if len(df) == 0:
        return "Please select d-code for its prediction"
    else:
        y = np.array(df[['Average Price-2022']]).reshape(-1, 1)
        X = np.array(df[['Date_Delta']]).reshape(-1, 1)
        return np.squeeze(LinearRegression().fit(X, y).predict(np.array(X_test).reshape(1, -1)))

def group_predictions(df, date):

    if len(df) == 0:
        return "Please select d-code for its prediction"
    else:
        
        date = pd.to_datetime(date)
        df.Date = pd.to_datetime(df.Date)

        day = np.timedelta64(1, 'D')
        mn = df.Date.min()
        df['Date_Delta'] = df.Date.sub(mn).div(day)

        dd = (date - mn) / day

        return df.groupby('DCode').apply(model, X_test=dd)


try:
    list_of_res = group_predictions(d_code_selection, date_predict)

    prediction_dcode = pd.DataFrame(list_of_res, columns=[f'predicted value for {date_predict}']).reset_index()
    prediction_dcode[f'predicted value for {date_predict}'] = prediction_dcode[f'predicted value for {date_predict}'].astype(float)
except:
    "Error in predicting D-Code"    

#----Predicting Credit df----

credit_selection['Date'] = pd.to_datetime(credit_selection['Date'])  
credit_selection['Date_Delta'] = (credit_selection['Date'] - credit_selection['Date'].min())  / np.timedelta64(1,'D')
#date_predict = '06-15-2022'

def model_credit(df, X_test):
    if len(df) == 0:
        return "Please select a Credit from drop-down menu for its prediction"
    else: 
        y = np.array(df[['Average Price']]).reshape(-1, 1)
        X = np.array(df[['Date_Delta']]).reshape(-1, 1)
        return np.squeeze(LinearRegression().fit(X, y).predict(np.array(X_test).reshape(1, -1)))

def group_predictions_credit(df, date):
    if len(df) == 0:
        return "Please select a Credit from drop-down menu for its prediction"
    else:
        date = pd.to_datetime(date)
        df.Date = pd.to_datetime(df.Date)

        day = np.timedelta64(1, 'D')
        mn = df.Date.min()
        df['Date_Delta'] = df.Date.sub(mn).div(day)

        dd = (date - mn) / day

        return df.groupby('Credit').apply(model_credit, X_test=dd)

try:
    list_of_res = group_predictions_credit(credit_selection, date_predict)
    prediction_credit = pd.DataFrame(list_of_res, columns=[f'predicted value for {date_predict}']).reset_index()
    prediction_credit[f'predicted value for {date_predict}'] = prediction_credit[f'predicted value for {date_predict}'].astype(float)
except:
    "Error in Predicting Credit"  


#-------------Printing----------------

left_column, right_column = st.columns(2)

with left_column:
    st.subheader(f'Predictions for Tomorrow - RIN')
    try:
        st.dataframe(prediction_dcode)
    except:
        "Please select a D-Code to predict"
    
with right_column:
    st.subheader(f'Predictions for Tomorrow - Credits')
    try:
        st.dataframe(prediction_credit)
    except:
        "Please select a Credit Name to predict"
    
    






# ---- Extra required --- 
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)



