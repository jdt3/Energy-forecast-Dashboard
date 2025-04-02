#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
print(os.getcwd())


# In[4]:


os.chdir('C:/Users/trive/Dashboard')


# In[6]:


print(os.getcwd())


# In[8]:


get_ipython().system('pip install dash==2.9.3  # or latest version')
get_ipython().system('pip install dash_bootstrap_components')


# In[9]:


import pandas as pd
import numpy as np


# In[10]:


from prophet import Prophet

model = Prophet()


# In[14]:


import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import dash_bootstrap_components as dbc


# In[16]:


# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server
app.title = "Energy Consumption Forecast" 


# In[18]:


# Load and process data
df = pd.read_csv('C:/Users/trive/Dashboard/Data/AEP_hourly.csv' , parse_dates=['Datetime'], index_col='Datetime')
print("Original data shape:", df.shape)


# In[20]:


# Resample to daily data
df_daily = df['AEP_MW'].resample('D').mean().dropna().reset_index()
df_daily.columns = ['ds', 'y']  # Prophet requires 'ds' and 'y' column names


# In[22]:


print("Daily data preview:")
print(df_daily.head())
print("Daily data shape:", df_daily.shape)


# In[24]:


# Initialize and train the model
model = Prophet()
model.fit(df_daily)

# Generate a future DataFrame for a 30-day forecast (or your desired horizon)
horizon = 30  # or another value
future = model.make_future_dataframe(periods=horizon)
forecast = model.predict(future)

print("Forecast preview:")
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())


# In[26]:


import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

# Initialize the Dash app (using JupyterDash if in a notebook)
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Energy Consumption Forecast Dashboard"

# Define the app layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Energy Consumption Forecast Dashboard"), width=12)
    ], className="my-2"),
    
    dbc.Row([
        dbc.Col([
            html.Label("Select Forecast Horizon (Days)"),
            dcc.Dropdown(
                id='horizon-dropdown',
                options=[
                    {'label': '30 Days', 'value': 30},
                    {'label': '90 Days', 'value': 90},
                    {'label': '365 Days (1 Year)', 'value': 365}
                ],
                value=30,
                clearable=False
            )
        ], width=3)
    ], className="my-2"),
    
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='forecast-graph')
        ], width=12)
    ]),
], fluid=True)

# Callback to update forecast graph based on selected horizon
@app.callback(
    Output('forecast-graph', 'figure'),
    Input('horizon-dropdown', 'value')
)
def update_forecast(horizon):
    # Train the Prophet model on the full daily data
    model = Prophet()
    model.fit(df_daily)
    
    # Create future dataframe and predict forecast
    future = model.make_future_dataframe(periods=horizon)
    forecast = model.predict(future)
    
    # Build the Plotly figure
    fig = go.Figure()
    
    # Historical data trace
    fig.add_trace(go.Scatter(
        x=df_daily['ds'],
        y=df_daily['y'],
        mode='lines',
        name='Historical Data'
    ))
    
    # Forecast trace
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat'],
        mode='lines',
        name='Forecast',
        line=dict(color='red')
    ))
    
    # Confidence interval traces
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat_lower'],
        fill=None,
        mode='lines',
        line=dict(color='lightpink'),
        name='Lower Confidence'
    ))
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat_upper'],
        fill='tonexty',
        mode='lines',
        line=dict(color='lightpink'),
        name='Upper Confidence'
    ))
    
    fig.update_layout(
        title=f"Forecast for {horizon} Days",
        xaxis_title="Date",
        yaxis_title="Energy Consumption (MW)",
        hovermode="x unified"
    )
    return fig

# Run the app (if in a script, otherwise use app.run_server(mode='inline') for JupyterDash)
if __name__ == '__main__':
    app.run(debug=True)


# In[ ]:




