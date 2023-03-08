import streamlit as st
import data_operator as do

theme_plotly = None # None or streamlit

df, df_inicial = do.carga_dados()

st.set_page_config(page_title='📊Exploring Data', page_icon=':bar_chart:', layout='wide')
st.title('📊 Exploratory Data Analytics')

with open('style.css')as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html = True)

f1, f2, f3, f4, f5 = st.columns(5)
variavel = f1.selectbox(
    "What variable do you want explorating?",
    ('Sales_Owner', 'Segment','Tax_Regime','Campaign','Company_Size'))


fig1, fig2, fig3, fig4, fig5 = do.analise_exploratoria(df, variavel)

st.plotly_chart(fig4,use_container_width=True)
a1, a2 = st.columns(2)
b1, b2 = st.columns(2)

a1.plotly_chart(fig1,use_container_width=True)
a2.plotly_chart(fig2,use_container_width=True)
b1.plotly_chart(fig3, use_container_width=True)
b2.plotly_chart(fig5, use_container_width=True)