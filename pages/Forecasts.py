import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
import modulos.data_operator as do

theme_plotly = None # None or streamlit
st.set_page_config(page_title='📉Forecasts📈', page_icon=':bar_chart:', layout='wide')
st.title('📉 Forecasts Of Funnel📈')


with open('style.css')as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html = True)

subtab_chart, subtab_table = st.tabs(['**Charts Forecasts**', '**Table / Indicators**'])
df, df_inicial = do.carga_dados()

df_previsoes = do.novas_previsoes(df)
df_final = do.concatena_formata(df_inicial,df_previsoes)
df_final = do.formata_previsoes(df_final)

with subtab_table:
    fi1, fi2, fi3, fi4 = st.columns(4)
    filtro1 = fi1.selectbox(
        "Select Sales Owner",
        ('All', 'Vendedor 01', 'Vendedor 03', 'Vendedor 04', 'Vendedor 02',
        'Gerente 02', 'Gerente 01', 'Analista'))
    filtro2 = fi2.selectbox(
        "Select Segment",
        ('All','Frigorifico', 'Agropecuaria', 'Distribuidora', 'Hortifrut',
        'Lanchonete', 'Loja de Departamento', 'Material de Construção',
        'Outros', 'Padaria', 'Resturante', 'Mercado'))
    filtro3 = fi3.selectbox(
        "Select Tax_Regime",
        ('All', 'Simples Nacional', 'Lucro Presumido', 'Lucro Real'))
    filtro4 = fi4.selectbox(
        "Select Campaign",
        ('All', 'prosp_merketing', 'Evento', 'google', 'indicacao'))

    # Tabela analitica
    tabela_previsoes,tot1, tot2, met_r, met_c, met_r2, met_c2, met_r3, met_c3, met_r4, met_c4 = do.tabela_previsao(df_final, filtro1, filtro2, filtro3, filtro4)
    ind1, ind2, ind3, ind4, ind5 = st.columns(5)
    ind1.metric(label="Total Monthly Revenue💲",value=tot1)
    ind2.metric(label="Monthly Revenue - Very High🟢",value=met_r)
    ind3.metric(label="Monthly Revenue - High🔵",value=met_r2)
    ind4.metric(label="Monthly Revenue - Medium🟡",value=met_r3)
    ind5.metric(label="Monthly Revenue - Low🔴",value=met_r4)
    ind6, ind7, ind8, ind9, ind10 = st.columns(5)
    ind6.metric(label="Total Prospects🪪",value=tot2)
    ind7.metric(label="Prospects - Very High🟢",value=met_c)
    ind8.metric(label="Prospects - High🔵",value=met_c2)
    ind9.metric(label="Prospects - Medium🟡",value=met_c3)
    ind10.metric(label="Prospects - Low🔴",value=met_c4)
    st.plotly_chart(tabela_previsoes, use_container_width=True)

with subtab_chart:
    f1, f2, f3= st.columns(3)
    variavel = f1.selectbox(
        "What variable do you want explorating?",
        ('Sales_Owner', 'Segment','Tax_Regime','Campaign'))
    
    options = f2.multiselect(
    'What indicator of closeing sales do you want?',
    ['Very High', 'High', 'Medium', 'Low'],
    ['Very High', 'High', 'Medium', 'Low'])

    if options == []:
        st.warning('You need to select at least one variable of the indicator close', icon="⚠️")
    else:
        fig1, fig2= do.analise_preditiva(df_final, variavel, options, options)
        a1, a2 = st.columns(2)
        a1.plotly_chart(fig1,use_container_width=True)
        a2.plotly_chart(fig2,use_container_width=True)