import pandas as pd
from datetime import date, datetime

from sklearn.preprocessing import StandardScaler

import plotly.express as px

import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
from dash import dash_table

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

import pathlib
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("data").resolve()

# importing datasets, merging and transforming
orders=pd.read_csv(DATA_PATH.joinpath("olist_orders_dataset.csv"))
customers=pd.read_csv(DATA_PATH.joinpath("olist_customers_dataset.csv"))
items=pd.read_csv(DATA_PATH.joinpath("olist_order_items_dataset.csv"))

df= orders.merge(customers,on="customer_id")
df= df.merge(items ,on="order_id")

df['order_purchase_timestamp'] = df['order_purchase_timestamp'].apply(lambda x:datetime.strptime(x,'%Y-%m-%d %H:%M:%S').date())

# prepare for clustering, filtering and scaling
dfc =df.groupby('customer_unique_id').agg({'order_id':'nunique', 'price':'sum', 'order_purchase_timestamp':lambda x: (date.today() - x.max()).days}).reset_index()  # .days atributte convert timedelta into int
dfc =dfc.rename(columns={'order_id':'frequency', 'price':'monetary', 'order_purchase_timestamp':'recency'})

df_last_purchase=df.groupby('customer_unique_id')['order_purchase_timestamp'].max().reset_index().rename(columns={'order_purchase_timestamp':'last_purchase'})
dfc = dfc.merge(df_last_purchase, on='customer_unique_id')
dfc=dfc.loc[dfc['last_purchase']>datetime.strptime('2018-08-01','%Y-%m-%d').date()]

dfcs=StandardScaler().fit_transform(dfc.drop(["customer_unique_id","last_purchase"], axis=1))


# navbar
navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Github", href='https://github.com/raqueldourado/Clustering-Olist-Clients',target='_blank'))
    ],
    brand="OLIST CLUSTERING - RECENCY FREQUENCY MONETARY (RFM)",
    color="#00008B",
    dark=True
)
# layout and callback

app.layout = html.Div([
    navbar,
    dbc.Row([
        dbc.Col([
            html.Label(['Choose the number of clusters:'], style={'font-weight': 'bold','margin-top': '2px'}),
            dcc.Dropdown([3,4,5,6], 4, id='k_dropdown', clearable=False, style={'width': '40%'}),
            dcc.Loading(
                id='loading_3d',
                type='default',
                children=[
                    dcc.Graph(id='3d_graph', style={'margin-top':'3px'}),
                    html.Div(id='s_score', style={'color':'#0000b3'})
                ]
            )
        ],style={'margin-left': '20px'}),
        dbc.Col([
            dcc.Loading(
                id='loading_table',
                type='default',
                children=[
                    html.Label(['Clusters info table'],style={'font-weight': 'bold','margin-top': '150px'}),
                    html.Div(
                        dash_table.DataTable(id='table1',
                                             columns=[{"name": 'cluster', "id": 'cluster'},
                                                      {"name": 'mean_frequency', "id": 'mean_frequency'},
                                                      {"name": 'mean_recency', "id": 'mean_recency'},
                                                      {"name": 'mean_monetary', "id": 'mean_monetary'},
                                                      {"name": 'cluster_size', "id": 'cluster_size'}]
                                             ),style={'margin-right': '70px'}
                    )
                ]
            )
        ])
    ])
])




@app.callback(
    Output('3d_graph', 'figure'),
    Output('s_score','children'),
    Output('table1', 'data'),
    Input('k_dropdown','value'),
)
def call1 (k):
    kmodel = KMeans(n_clusters=k, n_init=1, random_state=1)
    dfc["cluster"] = kmodel.fit_predict(dfcs)

    fig3d = px.scatter_3d(dfc, x='frequency', y='monetary', z='recency', color='cluster', opacity=0.5)
    fig3d.update_layout(margin=dict(l=0, r=0, b=0, t=30),title_text="3D View of clusters", title_x=0.5)
    fig3d.update_coloraxes(showscale=False)
    fig3d.update_traces(marker_size=3)

    ss = round(silhouette_score(dfcs, kmodel.labels_), 2)
    score_text = f" Mean Silhouette Coefficient : {ss}"

    table=dfc.groupby('cluster').agg(mean_frequency=pd.NamedAgg('frequency', 'mean'),
                                     mean_recency=pd.NamedAgg('recency', 'mean'),
                                     mean_monetary=pd.NamedAgg('monetary', 'mean'),
                                     cluster_size=pd.NamedAgg('cluster', 'count')).reset_index().round({"mean_frequency": 1, "mean_recency": 0, "mean_monetary": 0}).astype({'mean_recency': 'int', 'mean_monetary': 'int'})

    table_data = table.to_dict('records')

    return fig3d, score_text, table_data




if __name__ == '__main__':
    app.run_server(debug=True)