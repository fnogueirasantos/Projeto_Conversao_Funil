import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
import data_operator as do
import time

theme_plotly = None # None or streamlit
st.set_page_config(page_title='💡Simulations', layout='wide')
st.title('💡 Simulations')

with open('style.css')as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html = True)

i1, i2, i3, i4, i5, i6, i7 = st.columns(7)
input0 = 'Open'
input1 = i1.selectbox('Select Sales_Owner:',
                      ('Vendedor 01', 'Vendedor 02', 'Vendedor 04', 'Vendedor 03',
                      'Gerente 01', 'Gerente 02', 'Analista'))
input2 = i2.number_input("Select Value of Prospect:",5000)
input3 = i3.slider("Select number of stores:", 1, 50, 1)
input4 = i4.selectbox('Select Segment:',
                      ('Frigorifico', 'Agropecuaria', 'Distribuidora', 'Hortifrut',
                       'Lanchonete', 'Loja de Departamento', 'Material de Construção',
                       'Outros', 'Padaria', 'Resturante', 'Mercado', 'Vestuário'))
input5 = i5.selectbox('Select Tax_Regime:',
                      ('Lucro Presumido', 'Simples Nacional', 'Lucro Real'))
input6 = i6.selectbox('Select Campaign:',
                      ('prosp_merketing', 'google', 'Evento', 'indicacao'))
input7 = i7.selectbox('Select Company_Size:',
                      ('Pequeno', 'Médio', 'Grande'))
dados = [input0, input1, int(input2), int(input3), input4, input5, input6, input7]
nv_dados_input = pd.DataFrame(dados).T
nv_dados_input.rename(columns = {0:'Sales_Status',1:'Sales_Owner',2:'Value',
                                 3:'Store_Amount', 4:'Segment',
                                 5:'Tax_Regime',6:'Campaign',
                                 7:'Company_Size'}, inplace=True)
time.sleep(2)
a2, a3 = st.tabs(['****', '****'])
st.text('YOURS variables defined')
st.dataframe(nv_dados_input, use_container_width=True)
nv_dados_input['Start_Date'] = '05/09/2019'
nv_dados_input['Trading_Status'] = 'Proposta'
nv_dados_input['Id_Business'] = 'SIMULATION'

a2, a3 = st.tabs(['****', '****'])

if st.button('FORECAST'):
    time.sleep(1)
    df, df_inicial = do.carga_dados()
    df = pd.concat([df, nv_dados_input])
    df_inicial = pd.concat([df_inicial, nv_dados_input])
    df_previsoes = do.novas_previsoes(df)
    df_final = do.concatena_formata(df_inicial,df_previsoes)
    df_final = df_final.query('Id_Business == "SIMULATION"')
    df_final = do.formata_simulacoes(df_final)
    ind1, ind2, ind3, ind4 = st.columns(4)
    ind4, ind5, ind6, ind7 = st.columns(4)
    i_won = str(df_final['Prob_Won'].sum()).replace('.',',') + '%'
    ind2.metric(label="PROBABILITY WON✅",value=i_won)
    i_lost = str(df_final['Prob_Lost'].sum()).replace('.',',') + '%'
    ind3.metric(label="PROBABILITY LOST⛔",value=i_lost)
    indic = str(df_final['Indicator_Close'].min())
    ind2.metric(label="INDICATOR OF CLOSED📶",value=indic)
    indic2 = 'R$ ' + str(round(df_final['Monthly_Revenue'].sum(),2)).replace('.',',')
    ind3.metric(label="MONTHLY REVENUE💵",value=indic2)
    fig = do.grafico_simulations(df_final)
    ind1.plotly_chart(fig, use_container_width=True)

else:
    st.write('⚠️Click here to generate your forecast!!')