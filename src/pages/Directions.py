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

dash.register_page(__name__, path='/directions', name="Directions")


layout = html.Div([
    html.Div(className='card margin-200-top', children=[
        html.H3(children='Individual Shift Directions', className='card-title'),
        dcc.Markdown(className="info-box margin-50-bottom ", children=[
            '''
                Plot displays the gradients of abundance change for each species. It is calculated by partitioning the entire predicted area (see the area used in [ATI Maps](/ati)) of a species into a part where abundance increases (ATI > 0.5) and a part where abundance decreases (ATI < 0.5).
                The gradient is calculated as the difference in the average value of a variable between the two parts. As it is a difference, the value is given as ΔX. The magintude of these numbers can be interpreted as the strength of the gradient along a variable and compared across species.      
                
                **Important**: If, for example, ΔElevation = 300m for a species it does __not mean__ that the species shifted its range 300 m upwards. It merely means that the best conditions (to increase in abundance) are 300m above the worst conditions and that the species will experience an upwards "pull". The actual ranges will depend on many other factors, like land use, dispersal etc. and can't be inferred from these numbers. 
            
                **X / Y**: Select variables you want to compare.
                
                **Color**: Optional color coding of points by family, growth form etc. 
                
                **Dataset**: Defines how plots are used for model training. 
                - **Abundance Only** (65 species) uses only plots where the number of individuals decreased or increased, discarding those where it stayed the same. It is a stronger signal of range shifts.
                - **Abundance and DBH** (77 species), here plots with no change in abundance are included as well by comparing the total diameter at breast height (DBH). Pracitcally most plots (75%) with no abundance change end up getting an "increased" label. Gradients inferred from this datasets will point towards an increase in abundance OR survival of the species.     

                **Variable Set**: The set of variables used by the model. Updates what can be chosen on X and Y axis. The reliability of predictions is slightly higher for more variables, but the larger dataset has a large number of highly correlated and redundand variables. 
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