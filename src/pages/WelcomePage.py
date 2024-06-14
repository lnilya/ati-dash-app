import dash
import pandas as pd
from dash import html, callback, dcc, ctx
from dash.dependencies import Input, Output, State
import src.__libs.dashutil as dashutil
from src.classes import VariableList
from src.classes.Enums import ClassCombinationMethod, PredictiveVariableSet
from src.datautil import PlotProps
import plotly.graph_objects as go
import src.shiftpredutils as utils
import paths as PATHS
import plotly.express as px

dash.register_page(__name__, path='/', name="Welcome")

layout = html.Div([
    html.Div(className='card margin-200-top', children=[
        html.H3(children='Welcome', className='card-title'),
        ## SPECIES SCATTER
        html.Div(className='fl-row-start neg-margin-card card-section', children=[
            dcc.Markdown(className="welcome-box margin-50-bottom pad-50", children=[
                '''
                    Welcome to the ATI results overview web application. As a user, you have the power to study in detail the drivers, directions and locations of range shifts in New Zealand. This application is designed to empower you in your research, conservation, or environmental science work. 
                    
                    Getting started is easy. Simply click on any of the three buttons in the top right menu. Each page provides clear instructions, ensuring you feel confident and comfortable as you navigate through the application. 
                    
                    **Important things to remember**:
                    - Graphs will only display once all dropdown boxes have a value 
                    - Server responses can be delayed by a few seconds (the title of the tab in the browser changes to "Updating..." during the wait)
                    - Dataset choice (Abundance Only vs Abundance and DBH) will change the species available to select and interpret the ATI values. Abundance Only indicates fundamental abundance shifts; abundance and DBH, on the other hand, include survival, so positive ATI values here do not necessarily mean a range shift but merely survival.
                    Areas of decrease in abundance will be the same in both datasets. 
                    
                    **[Drivers](/drivers)**: Shows how important different variables are for predicting the abundance trend of different species.
                    
                    **[Maps](/maps)**: Generates a map of Abundance Trend Indicator Predictions (ATI) for New Zealand and shows where species are likely to increase or decrease in abundance.
                    
                    **[Directions](/directions)**: Displays the gradients of abundance change for each species and allows to see the breadth of individual species responses. 
                    
                    (Todo add intro video)  

                '''
            ]),
        ]),
    ])
])
