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

dash.register_page(__name__, path='/', name="Range Shifts")


layout = html.Div([
    html.Div(className='card margin-200-top', children=[
        html.H3(children='Plot Analysis Whole NZ', className='card-title'),
        dcc.Markdown(className="info-box margin-50-bottom ", children=[
            '''
                **Data**: Each point is belongs to a species and two consecutive measurements of the same plot. We look at how cover/height of this species in that plot changed. Open Access RECCE data entire NZ. 

                **Filters**: Either a specific growth form or a specific species.

                **X**: Different abiotic attributes of plots (location, elevation, slope etc)

                **Y**: Either Change in Cover or Height per year (between two consecutive measurements of the same plot)
            '''
        ]),

        ## SPECIES SCATTER
        html.Div(className='fl-row-start neg-margin-card card-section', children=[
            html.Div(className="col w-33", children=[
                dashutil.addDropDown("xsel", 'X-Axis:',
                                     [], persistence=True,
                                     multi=False, optionHeight=50, searchable=False),
                dashutil.addDropDown("comb", 'Dataset:', ClassCombinationMethod.getDict(),
                                     multi=False, optionHeight=50, searchable=False),
            ]),
            html.Div(className="col w-33", children=[
                dashutil.addDropDown("ysel", 'Y-Axis:', [], persistence=True,
                                     multi=False, optionHeight=50, searchable=False),
                dashutil.addDropDown("vars", 'Variable Set:',
                                     PredictiveVariableSet.getDict(),
                                     multi=False, optionHeight=50, searchable=False),

            ]),
            html.Div(className="col w-33", children=[
                dashutil.addDropDown("col", 'Color:',
                                 {"ThreatenedStatus":"Threatened Status",
                                      "Family":"Family",
                                      "BioStatus":"Bio Status",
                                      "GrowthForm":"Growth form"}, persistence=True,
                                     multi=False, optionHeight=50, searchable=False),
            ]),
        ]),
        html.Div(className='fl-row neg-margin-card card-section', children=[
            dcc.Graph(id="mainplot", style={'height': '600px'}, className='fl-grow'),
        ]),
    ])
])

@callback(
    Output("xsel", "options"),
    Output("ysel", "options"),
    Input("vars", "value"),
)
def setVars(comb):
    if comb is None or len(comb) == 0:
        return [],[]
    vl:VariableList = PredictiveVariableSet.fromString(comb)
    varDict = dict(zip(vl.list,PlotProps.renameBioClimToClearTextList(vl.list.copy())))
    return (varDict,varDict)

@callback(
    Output("mainplot", "figure"),
    Input("xsel", "value"),
    Input("ysel", "value"),
    Input("col", "value"),
    Input("comb", "value"),
    State("vars", "value"),
)
def drawPlot(x,y,col,comb,allVars):
    if x is None or y is None or comb is None or allVars is None or x == y:
        return go.Figure()


    vl:VariableList = PredictiveVariableSet.fromString(allVars)
    cc:ClassCombinationMethod = ClassCombinationMethod[comb]

    # "/Users/shabanil/Documents/Uni/DeslippeLab/ati-dash-app/"
    p = "assets/data/shiftdirections/Fig_ShiftDirections_Gradients_%s_%s.csv"%(vl.name,cc.name)

    df = pd.read_csv(p)
    f = utils.render2D(df, [x,y], col, showPlot=False)

    print("LOADED DATA")

    return f