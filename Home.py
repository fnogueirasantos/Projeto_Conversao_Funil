import streamlit as st
import data
import warnings
warnings.filterwarnings("ignore")

# Configuracao Pagina principal
theme_plotly = None # None or streamlit
st.set_page_config(page_title='🎯Sales Funnel - Home',  layout='wide')

with open('style.css')as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html = True)


t1, t2,t3 = st.columns((0.07,3.5,1)) 

t2.title("Web APP - Analysis of Sales Funnel and New Predictions")
t2.markdown("**Portfólio:** https://portfolio-production-c44b.up.railway.app/ **| Linkedin:** https://www.linkedin.com/in/felipenogueira92")

subtab_table = st.tabs(['***'])

df, df_inicial = data.carga_dados()

idioma = st.radio(
    "Idioma de introdução / Introduction Language:",
    ('English', 'Português'))

subtab_table = st.tabs(['***'])
if idioma == 'Português':
    st.title("Introdução")
else:
    st.title("Introduction")

if idioma == 'Português':
    t1, t2, t3, t4 = st.columns(4)
    with t1.expander("Problema de negócio"):
        st.write("""Aplicação web de dados criada com a finalidade de explorar e analisar os dados do funil de vendas,
                 entender os relacionamentos e com base nos dados históricos de negócios ganhos e 
                 perdidos, realizar as previsões de fechamento para o negócios com status "Open".
                 """)
    with t2.expander("Página 01: (Exploring Data)"):
        st.write(""" Automatizar a análise exploratoria das variáveis existentes por sua distribuição 
                em valor e quantidade. Criar um indicador de eficiência das conversões de leeds que atualiza 
                de acordo com a seleção da variável. 
                 """)
    with t3.expander("Página 02: (Forecasts)"):
        st.write("""Retornar as previsões usando machine learning com as probabilidades
                de ganho ou perda de todos os leeds com status de "Open". Elaborar um relatório de entrega
                dinâmico que permita o usuário realizar filtros e montar diversos cenários de acordo com as variáveis (vendedor, segmento, regime tributário e etc.)
                contendo gráficos, cartões de indicadores e uma tabela geral.
                 """)
    with t4.expander("Página 03: (Simulations)"):
        st.write(""" Criar um ambiente de simulação onde o usuário defina as variáveis de entrada e 
                receba as probabilidades de ganho ou perda. 
                 """)
else: 
    t1, t2, t3, t4 = st.columns(4)
    with t1.expander("Business Problem"):
        st.write("""Web aplication created with the object of explorating and analitycing the datas of sales funnel, 
                understand the relashonships and with the converts deals history datas, to make previsions for the deals with current status "Open".
                 """)
    with t2.expander("Page 01: (Exploring Data)"):
        st.write(""" To Automate of exploratory analysis of the variables for yours distribution and value. 
                Create indicator of effective that will be actualing with the filter select. 
                 """)
    with t3.expander("Page 02: (Forecasts)"):
        st.write("""It will be returned the predictions using machine learning with the probabilities of "Won" and "Lost" of the all deals with the current status "Open". 
                The finish report will be dynamic and will let the user to do many filters and to build scenarios with by the variables, also it will contain charts, cards and table.
                 """)
    with t4.expander("Page 03: (Simulations)"):
        st.write(""" Create an environment of simulation that the user choose/set entry variables and receive the probabilities of "Won" and "Lost". 
                 """)
        
if idioma == 'Português':
    with st.expander("Visualizar amostra dos dados brutos"):
        st.table(df_inicial.sample(20))
else:
    with st.expander("Visualize a sample of start data"):
        st.table(df_inicial.sample(20))