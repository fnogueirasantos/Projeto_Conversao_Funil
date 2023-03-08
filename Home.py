from datetime import datetime, timedelta
from datetime import date
from prophet import Prophet
from prophet.plot import plot_plotly
from pandas.io.json import json_normalize
import datetime
import streamlit as st
import pandas as pd
from pytrends.request import TrendReq
import plotly.graph_objects as go
import data_operator as do
import warnings
warnings.filterwarnings("ignore")

# Configuracao Pagina principal
st.set_page_config(page_title='🎯Marketing Funil - Analytics',  layout='wide')
t1, t2, = st.columns((0.07,1)) 
t2.title('Marketing Funil - Analytics')
c1, c2, c3 = st.columns(3)
with c1:
    st.info('**Perfil: [Felipe Nogueira](https://www.linkedin.com/in/felipenogueira92)**', icon="💡")
with c2:
    st.info('**GitHub: [@fnogueirasantos](https://github.com/fnogueirasantos)**', icon="💻")
with c3:
    st.info('**Data: [Flipside Crypto](https://flipsidecrypto.xyz)**', icon="🧠")




subtab_introduction, subtab_ml = st.tabs(['**Introduction**', '**Machine Learning**'])

