import dash
import pandas as pd
from dash import html, callback, dcc, ctx
from dash.dependencies import Input, Output,State
import src.__libs.dashutil as dashutil
from src.classes import VariableList
from src.classes.Enums import ClassCombinationMethod, PredictiveVariableSet
from src.datautil import PlotProps, getSpeciesInfo
import plotly.graph_objects as go
import src.shiftpredutils as utils
import paths as PATHS
import plotly.express as px

dash.register_page(__name__, path='/drivers-summary', name="Driver Summary")

layout = html.Div([
    html.Div(className='card margin-200-top', children=[
        html.H3(children='Drivers of Range Shifts', className='card-title'),
        dcc.Markdown(className="info-box margin-50-bottom ", children=[
            '''
                This section describes how important different variables are for predicting the abundance trend of different species.  
                It displays species on the X axis and the importance 
                
                **Dataset**: Defines how plots are used for model training. Species automatically update when changed. 
                - **Abundance Only** (65 species) uses only plots where the number of individuals decreased or increased, discarding those where it stayed the same. It is a stronger signal of range shifts.
                - **Abundance and DBH** (77 species): Here, plots with no change in abundance are included as well by comparing the total diameter at breast height (DBH). Practically, most plots (75%) with no abundance change end up getting an "increased" label. Positive correlations/importances indicate increased abundance __OR survival__ of the species.
                
                **Group Species**: Since there are too many species to get a good overview, they can be grouped by Family, Genus, Growth Form or Threatened Status.
                
                **Filter by Family**: Once you have selected the dataset, you can filter by family to only show species from a certain family. Multiple selections are possible.
                
                
                **Metric**: How drivers are identified. The sum of all values for each species will be normalized to 1 to get a relative importance of the driver for that species.
                - **RF Feature Importance**: The random forest generates Feature Importance values during training, which signify how vital a variable is for correct prediction. Here, large values indicate importance but not whether the species increase or decrease in abundance along this variable.
                - **Absolute Correlation**: This is the same as correlation, but the sign is discarded. It can be useful to compare the absolute relevance of multiple species to a variable.
                
                 
                **Exclude Variables**: To get a better overview you can exclude some variables from being displayed. Note that excluding variables will change the values displayed for the other variables, as the sum of all displayed values for each species is normalized to 1. 
                
                **Sorting**: Sorts the bars by the selected variable. The selected variable will appear at the top of the bar chart and species will be sorted in descending order 
            '''
        ]),


        ## SPECIES SCATTER
        html.Div(className='fl-row-start neg-margin-card card-section', children=[
            html.Div(className="col w-33", children=[
                dashutil.addDropDown("drvs_comb", 'Dataset:', ClassCombinationMethod.getDict(),
                                     initial="Abundance Only",
                                     multi=False, optionHeight=50, searchable=False),
                dashutil.addDropDown("drvs_group", 'Group Species:',
                {"Genus":"Genus","Family":"Family","GrowthForm":"Growth Form","ThreatenedStatus":"Threatened Status"},
                                     initial="Abundance Only",
                                     multi=False, optionHeight=50, searchable=False),
            ]),
            html.Div(className="col w-33", children=[
                dashutil.addDropDown("drvs_familyfilter", 'Filter by Family:',
                                     [], persistence=False,
                                     multi=True, optionHeight=50, searchable=True),
                dashutil.addDropDown("drvs_gffilter", 'Filter by Growth Form:',
                                     [], persistence=False,
                                     multi=True, optionHeight=50, searchable=True),
                dashutil.addDropDown("drvs_exvar", 'Exclude Variables:',
                                     dict(zip(PredictiveVariableSet.MinCorrelated.list,PlotProps.renameBioClimToClearTextList(PredictiveVariableSet.MinCorrelated.list.copy()))),
                                     persistence=True, initial="alpha",
                                     multi=True, optionHeight=50, searchable=False),
            ]),
            html.Div(className="col w-33", children=[
                dashutil.addDropDown("drvs_display", 'Metric (Y-Axis):',
                                     {"Absolute Correlation":"Absolute ATI Correlation", "Feature Importance":"RF Feature Importance"},
                                     persistence=True, initial="ATI Correlation",
                                     multi=False, optionHeight=50, searchable=True),

                dashutil.addDropDown("drvs_sort", 'Sorting (X-Axis):',
                                     dict(zip(PredictiveVariableSet.MinCorrelated.list,PlotProps.renameBioClimToClearTextList(PredictiveVariableSet.MinCorrelated.list.copy()))),
                                     persistence=True, initial="alpha",
                                     multi=False, optionHeight=50, searchable=False),
                html.Div(children="Alternatively: Click on a bar in the plot to sort by that variable", className="explanation"),
            ]),
        ]),
        html.Div(className='fl-row neg-margin-card card-section', children=[
            dcc.Graph(id="drvs_mainplot", style={'height': '600px'}, className='fl-grow'),
        ]),
    ])
])


@callback(
    Output("drvs_familyfilter", "options"),
    Output("drvs_gffilter", "options"),
    Input("drvs_comb", "value"),
)
def setVars(comb):
    if comb is None:
        return []

    df = pd.read_csv(PATHS.Results.fi(ClassCombinationMethod[comb]))

    # load species info and merge
    df, species = getSpeciesInfo(df, useCols=["Family","GrowthForm"])

    #get all families and their counts. Create a dict {family:family + "count"}
    familyCount = df.groupby("Family").size().reset_index(name="Count")
    familyCount["Label"] = familyCount["Family"] + " (" + familyCount["Count"].astype(str) + ")"
    fd = familyCount.set_index("Family").to_dict()["Label"]


    #Grwothform dict with counts
    gf = df.groupby("GrowthForm").size().reset_index(name="Count")
    gf["Label"] = gf["GrowthForm"] + " (" + gf["Count"].astype(str) + ")"
    gf = gf.set_index("GrowthForm").to_dict()["Label"]

    #sort both dicts alphabeticly by key
    fd = {k: fd[k] for k in sorted(fd)}
    gf = {k: gf[k] for k in sorted(gf)}


    return fd,gf

def _getVariableOrder(sortVar:str, renameBioClim:bool = True):
    # define a color for each variable
    varToCat = {varname: catlist[1] for varname, catlist in PlotProps.colors.items()}
    varToCol = {varname: catlist[0] for varname, catlist in PlotProps.colors.items()}
    varToLongName = {varname: lst[2] if len(lst) > 2 else varname for varname, lst in PlotProps.colors.items()}
    catOrder = ['Temperature', 'Geographic', 'Precipitation', 'Hydrology', 'Topography and Aspect', 'Soil', 'Wildlife',
                'Climate']

    # sort the variables by category
    def sort_variables(variables):
        return sorted(variables, key=lambda var: catOrder.index(varToCat[var]))

    varOrder = sort_variables(PredictiveVariableSet.MinCorrelated.list.copy())
    colorDict = {varToLongName[var]: varToCol[var] for var in varOrder}

    if sortVar is not None:
        varOrder = [v for v in varOrder if v != sortVar] + [sortVar]

    if renameBioClim:
        varOrder = PlotProps.renameBioClimToClearTextList(varOrder)
    return varOrder, colorDict
@callback(
    Output("drvs_mainplot", "figure"),
    Input("drvs_familyfilter", "value"),
    Input("drvs_gffilter", "value"),
    Input("drvs_group", "value"),
    Input("drvs_comb", "value"),
    Input("drvs_display", "value"),
    Input("drvs_sort", "value"),
    Input("drvs_exvar", "value"),
)
def drawPlot(filterFamily, filterGF, group, comb, disp, sort, exvar):
    if comb is None or disp is None:
        return go.Figure()

    if "Absolute Correlation" in disp:
        df = pd.read_csv(PATHS.Results.correlation(ClassCombinationMethod[comb]))
    elif "Feature Importance" in disp:
        df = pd.read_csv(PATHS.Results.fi(ClassCombinationMethod[comb]))

    #load species info and merge
    df,species = getSpeciesInfo(df,useCols=["Family","Genus","GrowthForm","ThreatenedStatus"])

    if filterFamily is not None and len(filterFamily) > 0:
        df = df[df.Family.isin(filterFamily)]
    if filterGF is not None and len(filterGF) > 0:
        df = df[df.GrowthForm.isin(filterGF)]

    #merge with df
    if group is not None:
        allGroups = list({"Species","Genus","Family","GrowthForm","ThreatenedStatus"} - {group})
        #drop the species column
        df = df.drop(columns=allGroups)
        #count how many members each species has in the group and make a df with group + count columns
        groupCount = df.groupby(group).size().reset_index(name="Count")
        #make a single column with the group name and the number in parentheses
        groupCount[group+"Count"] = groupCount[group] + " (" + groupCount["Count"].astype(str) + ")"
        #group by thge selected group and determine the mean
        df = df.groupby(group).mean().reset_index()

        #merge with groupCount to replace the group by groupCount
        df = df.merge(groupCount)

        #rename the group column to species
        df = df.rename(columns={group+"Count":"Species"})
        df.drop(columns=[group,"Count"], inplace = True)
        df = df.melt(id_vars=["Species"],var_name="Variable",value_name="Value")


    else:
        df = df.melt(id_vars=["Species","Genus", "Family", "GrowthForm", "ThreatenedStatus"],var_name="Variable",value_name="Value")

    #move variables to col
    # df = df.loc[df.Species.isin(filterFamily)]


    if disp == "Absolute Correlation":
        df["Value"] = df["Value"].abs()

    if exvar is not None:
        df = df[~df.Variable.isin(exvar)]


    #norm all to 1 to make stacked plot
    df["Value"] = df.groupby("Species")["Value"].transform(lambda x: x / x.sum())

    #define coloring and order of variable stacking

    varOrder, colorDict = _getVariableOrder(sort)

    speciesOrder = list(df.Species.unique())
    if sort is not None:
        speciesOrder = df[df.Variable == sort].sort_values("Value",ascending=False)["Species"]
    else:
        #sort species by alphabet
        speciesOrder.sort()

    df = PlotProps.renameBioClimToClearText(df,"Variable")
    f = px.bar(df, x="Species", y="Value",barmode="stack", category_orders={"Species":speciesOrder, "Variable":varOrder}, color="Variable",color_discrete_map=colorDict)

    #set Y label
    f.update_yaxes(title_text=disp + " (Sum normed to 1)")

    #make sure that the sorted variable appears at the bottom of the stack
    f.update_layout(legend=dict(traceorder='reversed'))

    #move legend to the top and make horizontal full width
    f.update_layout(legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="left",x=0),margin=dict(l=0,r=0,t=0,b=0))

    return f

#add a click handler for the main graph
@callback(
    Output("drvs_sort", "value"),
    Input("drvs_mainplot", "clickData"),
    State("drvs_mainplot", "figure"),
    State("drvs_sort", "value"),
    State("drvs_exvar", "value"),
)
def clickHandler(clickData, figure, sort, exVar):
    if clickData is None:
        return sort

    clickedVar = clickData["points"][0]["curveNumber"]
    vars, _ = _getVariableOrder(sort,False)
    #remove the exVar
    if exVar is not None:
        vars = [v for v in vars if v not in exVar]

    return vars[clickedVar]