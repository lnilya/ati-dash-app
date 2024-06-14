import dash
import pandas as pd
from dash import html, callback, dcc, ctx
from dash.dependencies import Input, Output,State
import src.__libs.dashutil as dashutil
from src.classes import VariableList
from src.classes.Enums import ClassCombinationMethod, PredictiveVariableSet
from src.datautil import PlotProps
import plotly.graph_objects as go
import src.shiftpredutils as utils
import paths as PATHS
import plotly.express as px

dash.register_page(__name__, path='/drivers', name="Drivers")


layout = html.Div([
    html.Div(className='card margin-200-top', children=[
        html.H3(children='Drivers of Range Shifts', className='card-title'),
        dcc.Markdown(className="info-box margin-50-bottom ", children=[
            '''
                This section describes how important different variables are for predicting the abundance trend of different species.  
                Select multiple species to compare them with one another. 
                
                **Dataset**: Defines how plots are used for model training. Species automatically update when changed. 
                - **Abundance Only** (65 species) uses only plots where the number of individuals decreased or increased, discarding those where it stayed the same. It is a stronger signal of range shifts.
                - **Abundance and DBH** (77 species): Here, plots with no change in abundance are included as well by comparing the total diameter at breast height (DBH). Practically, most plots (75%) with no abundance change end up getting an "increased" label. Positive correlations/importances indicate increased abundance __OR survival__ of the species.
                
                **Species**: Multiple species can be selected for comparison.
                
                **Metric**: How drivers are identified. 
                - **Correlation**: Correlation between predicted ATI values and a variable of interest. Here, a positive value indicates that an increase in this variable correlates with an increase in abundance (and additionally survival for the Abundance and DBH dataset). 
                - **Absolute Correlation**: This is the same as correlation, but the sign is discarded. It can be useful to compare the absolute relevance of multiple species to a variable. 
                - **RF Feature Importance**: The random forest generates Feature Importance values during training, which signify how vital a variable is for correct prediction. Here, large values indicate importance but not whether the species increase or decrease in abundance along this variable.
                
                **Sorting**: Defines in which order the variables are displayed on the X-axis.
            '''
        ]),

        ## SPECIES SCATTER
        html.Div(className='fl-row-start neg-margin-card card-section', children=[
            html.Div(className="col w-33", children=[
                dashutil.addDropDown("drv_comb", 'Dataset:', ClassCombinationMethod.getDict(),
                                     initial="Abundance Only",
                                     multi=False, optionHeight=50, searchable=False),
            ]),
            html.Div(className="col w-33", children=[
                dashutil.addDropDown("drv_species", 'Species:',
                                     [], persistence=False,
                                     multi=True, optionHeight=50, searchable=True),
            ]),
            html.Div(className="col w-33", children=[
                dashutil.addDropDown("drv_display", 'Metric (Y-Axis):',
                                     {"Correlation":"ATI Correlation","Absolute Correlation":"Absolute ATI Correlation", "Feature Importance":"RF Feature Importance"},
                                     persistence=True, initial="ATI Correlation",
                                     multi=False, optionHeight=50, searchable=True),
                dashutil.addDropDown("drv_sort", 'Sorting (X-Axis):',
                                     {"alpha":"By Alphabet","dif":"Largest average difference", "avg":"Largest average value", "avgabs":"Largest absolute average value"},
                                     persistence=True, initial="alpha",
                                     multi=False, optionHeight=50, searchable=False),
            ]),
        ]),
        html.Div(className='fl-row neg-margin-card card-section', children=[
            dcc.Graph(id="drv_mainplot", style={'height': '600px'}, className='fl-grow'),
        ]),
    ])
])


@callback(
    Output("drv_species", "options"),
    Input("drv_comb", "value"),
)
def setVars(comb):
    if comb is None:
        return []


    df = pd.read_csv(PATHS.Results.correlation(ClassCombinationMethod[comb]))
    sp = list(df.Species)
    sp.sort()
    return sp

@callback(
    Output("drv_mainplot", "figure"),
    Input("drv_species", "value"),
    Input("drv_comb", "value"),
    Input("drv_display", "value"),
    Input("drv_sort", "value"),
)
def drawPlot(species,comb,disp, sort):
    if species is None or comb is None or disp is None:
        return go.Figure()

    if "Correlation" in disp:
        df = pd.read_csv(PATHS.Results.correlation(ClassCombinationMethod[comb]))
    elif "Feature Importance" in disp:
        df = pd.read_csv(PATHS.Results.fi(ClassCombinationMethod[comb]))

    #move variables to col
    df = df.loc[df.Species.isin(species)]
    df = df.melt(id_vars=["Species"],var_name="Variable",value_name="Value")
    df = PlotProps.renameBioClimToClearText(df,"Variable")

    if disp == "Absolute Correlation":
        df["Value"] = df["Value"].abs()

    if sort == "alpha":
        order = list(df.Variable.unique())
        order.sort()
    elif sort == "avg":
        dfo = df.groupby("Variable").mean(numeric_only=True).reset_index()
        order = list(dfo.sort_values("Value",ascending=False)["Variable"])
    elif sort == "avgabs":
        dfo = df.groupby("Variable").mean(numeric_only=True).reset_index()
        dfo.Value = dfo.Value.abs()
        order = list(dfo.sort_values("Value",ascending=False)["Variable"])
    elif sort == "dif":
        dfoMax = df.groupby("Variable").max(numeric_only=True).reset_index()
        dfoMin = df.groupby("Variable").min(numeric_only=True).reset_index()
        dfoMax["Dif"] = dfoMax.Value - dfoMin.Value
        order = list(dfoMax.sort_values("Dif",ascending=False)["Variable"])

    f = px.bar(df, x="Variable", y="Value", color="Species", barmode="group", category_orders={"Variable":order})

    #set Y label
    f.update_yaxes(title_text=disp)

    return f