import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier  
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.preprocessing import LabelEncoder
import plotly.graph_objects as go
import plotly.express as px
import locale
locale.setlocale(locale.LC_MONETARY, 'pt_BR.UTF-8')
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("dados.csv", sep = ";")
df_inicial = df.copy()

def carga_dados():
	df = pd.read_csv("dados.csv", sep = ";")
	df_inicial = df.copy()
	return(df, df_inicial)


# Organizacao dos dados
def cria_modelo(df):
	# Removendo a coluna Id_Business, Start_Date e Trading_Status e fitrando os dados de negócios com status de "won" ou "lost"
	df.drop(columns=['Start_Date','Trading_Status','Id_Business'], inplace = True)
	df = df.query('Sales_Status != "Open"')	
	lb = LabelEncoder()
	df_modelo = df.copy()
	lista_categorica = df_modelo.select_dtypes(include = "object").columns
	lista_categorica
	for x in lista_categorica:
		df_modelo[x] = lb.fit_transform(df_modelo[x].astype(str))
	atributos = [ft for ft in list(df_modelo) if ft not in ['Sales_Status']]
	target = 'Sales_Status'
	X_treino, X_teste, y_treino, y_teste = train_test_split(df_modelo[atributos],
                                                        df_modelo[target], 
                                                        test_size = 0.2, 
                                                        random_state = 1)

	atributos = [ft for ft in list(df_modelo) if ft not in ['Sales_Status']]
	target = 'Sales_Status'
	X_treino, X_teste, y_treino, y_teste = train_test_split(df_modelo[atributos],
                                                        df_modelo[target], 
                                                        test_size = 0.2, 
                                                        random_state = 1)
	modelo = GradientBoostingClassifier(learning_rate = 0.1, max_depth = 10, random_state = 10)
	modelo.fit(X_treino[atributos], y_treino)
	previsoes = modelo.predict_proba(X_teste)
	best_preds = np.asarray([np.argmax(line) for line in previsoes])

	return (modelo,X_treino, X_teste, y_treino, y_teste, best_preds)

def novas_previsoes(df):
	df_novos = df.query('Sales_Status == "Open"')
	df_novos.drop(columns=['Start_Date','Trading_Status','Id_Business'], inplace=True)
	lista_categorica2 = df_novos.select_dtypes(include = "object").columns
	lb = LabelEncoder()
	for x in lista_categorica2:
		df_novos[x] = lb.fit_transform(df_novos[x].astype(str))
	atributos = [ft for ft in list(df_novos) if ft not in ['Sales_Status']]
	modelo,X_treino, X_teste, y_treino, y_teste, best_preds = cria_modelo(df)
	novos_dados = pd.DataFrame(df_novos[atributos], 
			    columns = X_teste.columns.values)
	previsoes_novos_dados = modelo.predict_proba((novos_dados))
	df_previsoes = pd.DataFrame(previsoes_novos_dados)
	df_previsoes.insert(0, column = "ID", value = range(1, 1 + len(df_previsoes)))

	return(df_previsoes)
	
def concatena_formata(df_inicial,df_previsoes):
	df_novos = df_inicial.query('Sales_Status == "Open"')
	df_novos.insert(0, column = "ID", value = range(1, 1 + len(df_novos)))
	df_final = pd.merge(df_previsoes, df_novos, on = 'ID', how = 'inner')

	return(df_final)

def formata_previsoes(df):
	df.drop(columns=['Start_Date','Sales_Status'], inplace = True)
	df.rename(columns={0:'Prob_Lost',1:'Prob_Won','Value':'Annual_Revenue'}, inplace=True)
	df['Annual_Revenue'] = round(df['Annual_Revenue'],2)
	df['Prob_Lost'] = round(df['Prob_Lost']*100,2)
	df['Prob_Won'] = round(df['Prob_Won']*100,2)
	df['Monthly_Revenue'] = round(df['Annual_Revenue'] / 12,2)
	df['Indicator_Close'] = ['Very High' if x >= 75 else 'High' if x >= 50 else 'Medium' if x >= 35 else 'Low' for x in df['Prob_Won']]
	df = df[['Id_Business','Sales_Owner','Prob_Won','Prob_Lost','Trading_Status',
	  'Annual_Revenue','Monthly_Revenue','Segment','Tax_Regime','Campaign','Indicator_Close']]

	return(df)

def formata_simulacoes(df):
	df.drop(columns=['Start_Date','Sales_Status'], inplace = True)
	df.rename(columns={0:'Prob_Lost',1:'Prob_Won','Value':'Annual_Revenue'}, inplace=True)
	df['Prob_Lost'] = round(df['Prob_Lost']*100,2)
	df['Prob_Won'] = round(df['Prob_Won']*100,2)
	df['Monthly_Revenue'] = df['Annual_Revenue'] / 12
	df['Indicator_Close'] = ['Very High' if x >= 75 else 'High' if x >= 50 else 'Medium' if x >= 35 else 'Low' for x in df['Prob_Won']]
	df = df[['Id_Business','Sales_Owner','Prob_Won','Prob_Lost','Trading_Status',
	  'Annual_Revenue','Monthly_Revenue','Segment','Tax_Regime','Campaign','Indicator_Close']]

	return(df)

def tabela_previsao(df, filtro1, filtro2, filtro3, filtro4):
	if filtro1 == 'All':
		df = df
	else:
		df = df.query(f'Sales_Owner =="{filtro1}"')
	if filtro2 == 'All':
		df = df
	else:
		df = df.query(f'Segment =="{filtro2}"')
	if filtro3 == 'All':
		df = df
	else:
		df = df.query(f'Tax_Regime =="{filtro3}"')
	if filtro4 == 'All':
		df = df
	else:
		df = df.query(f'Campaign =="{filtro4}"')
			
	fig = go.Figure(data=[go.Table(
		header=dict(values=list(df.columns),
	    	font=dict(size=12, color = 'white'),
            fill_color = '#264653',
	    	line_color = 'darkslategray',
            align=['left', 'center']),
		cells=dict(values=[df[x].tolist() for x in df.columns],
	       	fill_color='#014653',
            align=['left', 'center'],
			height=20))
			])
	fig.update_layout(title_text="Analytic table of the probabilities",
		   title_font_color = '#264653',
		   title_x=0,margin= dict(l=0,r=10,b=10,t=30), height=480)
	
	met = df.copy()
	tot1 = locale.currency(round(met['Monthly_Revenue'].sum(),2))
	tot2 = met['Monthly_Revenue'].count()
	met2 = df.query('Indicator_Close == "Very High"')
	met_r = locale.currency(round(met2['Monthly_Revenue'].sum(),2))
	met_c = met2['Monthly_Revenue'].count()
	met3 = df.query('Indicator_Close == "High"')
	met_r2 = locale.currency(round(met3['Monthly_Revenue'].sum(),2))
	met_c2 = met3['Monthly_Revenue'].count()
	met4 = df.query('Indicator_Close == "Medium"')
	met_r3 = locale.currency(round(met4['Monthly_Revenue'].sum(),2))
	met_c3 = met4['Monthly_Revenue'].count()
	met5 = df.query('Indicator_Close == "Low"')
	met_r4 = locale.currency(round(met5['Monthly_Revenue'].sum(),2))
	met_c4 = met5['Monthly_Revenue'].count()
	
	return(fig, tot1, tot2, met_r, met_c, met_r2, met_c2, met_r3, met_c3, met_r4, met_c4)

def analise_exploratoria(df, variavel):

	#Grafico 01
	dados_won = df.query('Sales_Status == "Won"')
	dados_won = pd.DataFrame(dados_won.groupby([variavel]).sum()['Value'].sort_values(ascending=False)).reset_index()
	dados_lost = df.query('Sales_Status == "Lost"')
	dados_lost = pd.DataFrame(dados_lost.groupby([variavel]).sum()['Value'].sort_values(ascending=False)).reset_index()
	trace1g1 = go.Bar(
		x=dados_won[variavel],
		y=dados_won['Value'],
		name = 'Won',
		marker=dict(color='#32cd7f')
	)
	trace2g1 = go.Bar(
		x=dados_lost[variavel],
		y=dados_lost['Value'],
		name = 'Lost',
		marker=dict(color='#C70039')
	)
	data1 = [trace1g1, trace2g1]

	layout1 = go.Layout(
		title=f'Funil Convert - Annual Revenue / By {variavel}',
		barmode='stack',
		margin= dict(l=0,r=10,b=10,t=30),
		width=280
	)
	fig1 = go.Figure(data=data1, layout=layout1)

	# Tabela
	dados_won2 = df.query('Sales_Status == "Won"')
	dados_won2 = pd.DataFrame(dados_won2.groupby([variavel])['Id_Business'].count().reset_index().rename(columns={'Id_Business':'Qtd_Won'}))
	dados_won2['%Qtd_Won'] = round(100 * dados_won2['Qtd_Won'] / dados_won2['Qtd_Won'].sum(),2)
	dados_lost2 = df.query('Sales_Status == "Lost"')
	dados_lost2 = pd.DataFrame(dados_lost2.groupby([variavel])['Id_Business'].count().reset_index().rename(columns={'Id_Business':'Qtd_Lost'}))
	dados_lost2['%Qtd_Lost'] = round(100 * dados_lost2['Qtd_Lost'] / dados_lost2['Qtd_Lost'].sum(),2)
	df_won_lost2 = pd.merge(dados_won2,dados_lost2, how='inner')
	df_won_lost2['%_Effective'] = round((df_won_lost2['Qtd_Won'] / (df_won_lost2['Qtd_Won'] + df_won_lost2['Qtd_Lost']))*100,2)
	df_won_lost2['%_Ineffective'] = round(100-df_won_lost2['%_Effective'],2)
	dados_result_won = df.query('Sales_Status == "Won"')
	dados_result_won = pd.DataFrame(dados_result_won.groupby([variavel]).sum('Value').reset_index().rename(columns={'Value':'Value_Won'}))
	dados_result_won.drop(columns=['Store_Amount'],inplace = True)
	dados_result_won['%Value_Won'] = round(100 * dados_result_won['Value_Won'] / dados_result_won['Value_Won'].sum(),2)
	dados_result_lost = df.query('Sales_Status == "Lost"')
	dados_result_lost = pd.DataFrame(dados_result_lost.groupby([variavel]).sum('Value').reset_index().rename(columns={'Value':'Value_Lost'}))
	dados_result_lost.drop(columns=['Store_Amount'],inplace = True)
	dados_result_lost['%Value_Lost'] = round(100 * dados_result_lost['Value_Lost'] / dados_result_lost['Value_Lost'].sum(),2)
	dados_result_Won_lost = pd.merge(dados_result_won,dados_result_lost, how='inner')
	dados_result_Won_lost['Value_Won'] = round(dados_result_Won_lost['Value_Won'],2)
	dados_result_Won_lost['Value_Lost'] = round(dados_result_Won_lost['Value_Lost'],2)
	df_won_lost2 =  pd.merge(dados_result_Won_lost,df_won_lost2, how='inner')

	fig4 = go.Figure(data=[go.Table(
		header=dict(values=list(df_won_lost2.columns),
	    	font=dict(size=12, color = 'white'),
            fill_color = '#264653',
	    	line_color = 'darkslategray',
            align=['left', 'center']),
		cells=dict(values=[df_won_lost2[x].tolist() for x in df_won_lost2.columns],
	       	fill_color='#014653',
            align=['left', 'center'],
			height=20,
			))
		])
	fig4.update_layout(title_text="Overview Table",
		   title_font_color = 'white',
		   title_x=0, margin= dict(l=0,r=10,b=10,t=30), 
		   height=180, width=1000
		   )

	tabela = df_won_lost2.copy()

	#Grafico 02
	trace1g2 = go.Bar(
		x=tabela['Qtd_Won'],
		y=tabela[variavel],	
		name = 'Won',
		orientation = "h",
		marker=dict(color='#32cd7f')
	)
	trace2g2 = go.Bar(
		x=tabela['Qtd_Lost'],
		y=tabela[variavel],
		orientation = "h",
		name = 'Lost',
		marker=dict(color='#C70039')
	)
	data2 = [trace1g2, trace2g2]

	layout2 = go.Layout(
		title=f'Funil Convert - Count / By {variavel}',
		barmode='stack',
		margin= dict(l=0,r=10,b=10,t=30),
		width=460
	)
	fig2 = go.Figure(data=data2, layout=layout2)

	#Grafico 03
	trace1g3 = go.Bar(
		x=tabela['%_Effective'],
		y=tabela[variavel],
		name = 'Won',
		orientation = "h",
		marker=dict(color='#32cd7f')
	)
	trace2g3 = go.Bar(
		x=tabela['%_Ineffective'],
		y=tabela[variavel],
		name = 'Lost',
		orientation = "h",
		marker=dict(color='#C70039')
	)
	data3 = [trace1g3,trace2g3]

	layout3 = go.Layout(
		title=f'Funil Convert - % Off Effective x Ineffective / By {variavel}',
		barmode='stack',
		margin= dict(l=0,r=10,b=10,t=30),
		width=460
	)
	fig3 = go.Figure(data=data3, layout=layout3)

	#Grafico 04
	trace0g4 = go.Scatter(
    x = dados_result_Won_lost[variavel],
    y = dados_result_Won_lost['%Value_Won'],
    mode = 'markers',
    name = 'Won',
    marker = dict(color = 'seagreen',
                  size = 15, showscale = False))

	trace1g4 = go.Scatter(
		x = dados_result_Won_lost[variavel],
		y = dados_result_Won_lost['%Value_Lost'],
		mode = 'markers',
		name = 'Lost',
		marker = dict(color = 'indianred',
					size = 15, showscale = False))
	data4 = [trace0g4, trace1g4]  # assign traces to data
	layout4 = go.Layout(
		title = f'% de Leeds Lost X Won off Total / By {variavel}',
		margin= dict(l=0,r=10,b=10,t=30)
	)
	fig5 = go.Figure(data=data4,layout=layout4)
	
	return(fig1, fig2, fig3, fig4, fig5)

def analise_preditiva(df, variavel, filtro, filtro2):

	#Grafico 01
	dados_very_high = df.query('Indicator_Close == "Very High"')
	dados_very_high = pd.DataFrame(dados_very_high.groupby([variavel]).sum()['Annual_Revenue'].sort_values(ascending=False)).reset_index()
	dados_high = df.query('Indicator_Close == "High"')
	dados_high = pd.DataFrame(dados_high.groupby([variavel]).sum()['Annual_Revenue'].sort_values(ascending=False)).reset_index()
	dados_medium = df.query('Indicator_Close == "Medium"')
	dados_medium = pd.DataFrame(dados_medium.groupby([variavel]).sum()['Annual_Revenue'].sort_values(ascending=False)).reset_index()
	dados_low = df.query('Indicator_Close == "Low"')
	dados_low = pd.DataFrame(dados_low.groupby([variavel]).sum()['Annual_Revenue'].sort_values(ascending=False)).reset_index()

	trace1g1 = go.Bar(
		x=dados_very_high[variavel],
		y=dados_very_high['Annual_Revenue'],
		name = 'Very High',
		marker=dict(color='#32cd7f')
	)
	trace2g1 = go.Bar(
		x=dados_high[variavel],
		y=dados_high['Annual_Revenue'],
		name = 'High',
		marker=dict(color='#00b5cd')
	)
	trace3g1 = go.Bar(
		x=dados_medium[variavel],
		y=dados_medium['Annual_Revenue'],
		name = 'Medium',
		marker=dict(color='#f1c232')
	)
	trace4g1 = go.Bar(
		x=dados_low[variavel],
		y=dados_low['Annual_Revenue'],
		name = 'Low',
		marker=dict(color='#C70039')
	)
	filtro = pd.DataFrame(filtro)
	d = {'Very High':trace1g1, 'High':trace2g1, 'Medium':trace3g1, 'Low':trace4g1}
	filtro[0] = filtro[0].map(d)
	filtro = list(filtro[0])

	layout1 = go.Layout(
		title=f'Forecast Funil Convert - Annual Revenue / By {variavel}',
		barmode='stack',
		margin= dict(l=0,r=10,b=10,t=30),
		width=280
	)
	fig1 = go.Figure(data=filtro, layout=layout1)

	#Grafico 02
	dados_very_high2 = df.query('Indicator_Close == "Very High"')
	dados_very_high2 = pd.DataFrame(dados_very_high2.groupby([variavel]).count().reset_index())
	dados_high2 = df.query('Indicator_Close == "High"')
	dados_high2 =pd.DataFrame(dados_high2.groupby([variavel]).count().reset_index())
	dados_medium2 = df.query('Indicator_Close == "Medium"')
	dados_medium2 = pd.DataFrame(dados_medium2.groupby([variavel]).count().reset_index())
	dados_low2 = df.query('Indicator_Close == "Low"')
	dados_low2 = pd.DataFrame(dados_low2.groupby([variavel]).count().reset_index())

	trace1g2 = go.Bar(
		y=dados_very_high2[variavel],
		orientation = "h",
		x=dados_very_high2['Annual_Revenue'],
		name = 'Very High',
		marker=dict(color='#32cd7f')
	)
	trace2g2 = go.Bar(
		y=dados_high2[variavel],
		orientation = "h",
		x=dados_high2['Annual_Revenue'],
		name = 'High',
		marker=dict(color='#00b5cd')
	)
	trace3g2 = go.Bar(
		y=dados_medium2[variavel],
		orientation = "h",
		x=dados_medium2['Annual_Revenue'],
		name = 'Medium',
		marker=dict(color='#f1c232')
	)
	trace4g2 = go.Bar(
		y=dados_low2[variavel],
		orientation = "h",
		x=dados_low2['Annual_Revenue'],
		name = 'Low',
		marker=dict(color='#C70039')
	)
	filtro2 = pd.DataFrame(filtro2)
	d2 = {'Very High':trace1g2, 'High':trace2g2, 'Medium':trace3g2, 'Low':trace4g2}
	filtro2[0] = filtro2[0].map(d2)
	filtro2 = list(filtro2[0])

	layout2 = go.Layout(
		title=f'Forecast Funil Convert - Count / By {variavel}',
		barmode='stack',
		margin= dict(l=0,r=10,b=10,t=30),
		width=280
	)
	fig2 = go.Figure(data=filtro2, layout=layout2)

	return(fig1, fig2)

def grafico_simulations(df):
	dfprob_won = df[['Prob_Won']]
	dfprob_won.rename(columns={'Prob_Won': 'Probability'}, inplace=True)
	dfprob_won['Label'] = 'Won'
	dfprob_lost = df[['Prob_Lost']]
	dfprob_lost.rename(columns={'Prob_Lost': 'Probability'}, inplace=True)
	dfprob_lost['Label'] = 'Lost'
	df_final = pd.concat([dfprob_won, dfprob_lost])
	cores = ['#0c7a1e','#bf0f24']
	fig = go.Figure([go.Pie(labels = df_final['Label'], 
			values = df_final['Probability'], hole = 0.3,
			marker=dict(colors=cores)
			#color_discrete_sequence=px.colors.sequential.RdBu
				)])

	fig.update_traces(hoverinfo = 'label+percent', 
                  textinfo = 'percent', 
                  textfont_size = 12)

	fig.update_layout(title = "Probability to convert leed (Lost X Won)", title_x = 0.5)

	return(fig)
