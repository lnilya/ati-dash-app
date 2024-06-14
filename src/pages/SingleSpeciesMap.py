import dash
import plotly.graph_objects as go
from dash import html, callback, dcc
from dash.dependencies import Input, Output, State

import paths as PATHS
import src.__libs.dashutil as dashutil
from GlobalParams import GlobalParams
from src.__libs.osutil import dirutil
from src.classes import ModelMeanPrediction, SimilarityData
from src.classes.Enums import ClassCombinationMethod, ModelType, PredictiveVariableSet, ClassificationProblem, \
    NoiseRemovalBalance
from src.classes.FileIDClasses import ModelFileID, ModelMeanPredictionFileID, SimilarityDataFileID
from src.classes.ModelMeanPrediction import ModelMeanPrediction

import plotly.express as px

from src.classes.SimilarityData import SimilarityData

dash.register_page(__name__, path='/ati', name="ATI Maps")


layout = html.Div([
    html.Div(className='card margin-200-top', children=[
        html.H3(children='ATI Maps', className='card-title'),
        dcc.Markdown(className="info-box margin-50-bottom ", children=[
            '''
                Shows the ATI predictions on a map. The ATI is a value between 0 and 1 that indicates where a species likely increased (>0.5 in red) or decreased (<0.5 in blue) in abundance. The data to generate these maps is an average over the years 1969-2019 and reflects past movements, not future projections. 

                **Dataset**: Defines how plots are used for model training. Species automatically update when changed. 
                - **Abundance Only** (65 species) uses only plots where the number of individuals decreased or increased, discarding those where it stayed the same. It is a stronger signal of range shifts.
                - **Abundance and DBH** (77 species): Here, plots with no change in abundance are included as well by comparing the total diameter at breast height (DBH). Practically, most plots (75%) with no abundance change end up getting an “increased” label. Therefore, values >0.5 in this dataset indicate an increase in abundance OR the survival of the species.     
                
                **Species**: Updates with choice of dataset.  
                
                **Area**: To avoid using the models for extrapolation outside the climatic conditions of the training data, we use a similarity metric to limit the area where predictions are made. 
                It compares conditions in each pixel with the conditions of the training data using the nearest neighbour average (see paper for details). 
                - **95% Occurrence Area (S') **: The range of a species is defined as an area containing 95% of occurrences (S' in paper). It is the most conservative estimate of the area in the model and is trusted to make reliable predictions. 
                - **Default (S) **: Predictions right outside the bounds are valuable and necessary to see where the species will expand to, so the default setting is an area slightly larger than the 95% occurrence interval and is called S in the paper.
                - **No Limit **: Predictions are made without any restriction. This is useful to see how the predictions diverge from the training data. However, these predictions should not be trusted. This setting is for visualisation purposes only. 
                 
            '''
        ]),

        ## SPECIES SCATTER
        html.Div(className='fl-row-start neg-margin-card card-section', children=[
            html.Div(className="col w-33", children=[
                dashutil.addDropDown("ssm_comb", 'Dataset:', ClassCombinationMethod.getDict(),
                                     initial="Abundance Only",
                                     multi=False, optionHeight=50, searchable=False),
            ]),
            html.Div(className="col w-33", children=[
                dashutil.addDropDown("ssm_species", 'Species:',
                                 [], persistence=False,
                                     multi=False, optionHeight=50, searchable=True),
            ]),
            html.Div(className="col w-33", children=[
                dashutil.addDropDown("ssm_similarity", 'Area:',
                                     {"1.2":"Default (S)", "1":"Only 95% of occurrences (S')", "100000":"No Limit (predictions unreliable)" },
                                     initial=1.2,
                                     persistence=True,multi=False, optionHeight=50, searchable=False),
            ]),
        ]),
        html.Div(className='fl-row neg-margin-card card-section', children=[
            dcc.Graph(id="ssm_mainplot", style={'height': '1000px'}, className='fl-grow'),
        ]),
    ])
])

@callback(
    Output("ssm_species", "options"),
    Input("ssm_comb", "value"),
)
def setVars(comb):
    if comb is None:
        return []

    nr = comb == "AdultsOnly" and 0.1 or 0.15
    mid = ModelFileID(None, ModelType.RFEnsemble, PredictiveVariableSet.Full,
                      ClassificationProblem.IncDec, ClassCombinationMethod(comb), nr, NoiseRemovalBalance.Equal)

    mfid = ModelMeanPredictionFileID(GlobalParams.yearRange,mid)
    allF = mfid.getAllFiles()
    allSpecies = [v[0]  for k,v in allF[0].items()]

    allSpecies.sort()
    return allSpecies

@callback(
    Output("ssm_mainplot", "figure"),
    Input("ssm_species", "value"),
    Input("ssm_similarity", "value"),
    Input("ssm_comb", "value"),
)
def drawPlot(sp,simcut,comb):
    if sp is None or simcut is None or comb is None:
        return go.Figure()

    nr = comb == "AdultsOnly" and 0.1 or 0.15
    mid = ModelFileID(sp, ModelType.RFEnsemble, PredictiveVariableSet.Full,
                      ClassificationProblem.IncDec, ClassCombinationMethod(comb), nr, NoiseRemovalBalance.Equal)

    mfid = ModelMeanPredictionFileID(GlobalParams.yearRange, mid)
    mmp = ModelMeanPrediction.load(mfid.file)

    smid = SimilarityDataFileID(GlobalParams.yearRange, GlobalParams.similarity_k, GlobalParams.similarity_metric, sp, PredictiveVariableSet.Full,
                                ClassificationProblem.IncDec, ClassCombinationMethod(comb))
    sm = SimilarityData.load(smid.file)

    mask2D= sm.getMask(float(simcut))
    k = 0

    pred = mmp.getDataAs2DArray(mask2D)
    f = px.imshow(pred, color_continuous_scale=px.colors.sequential.RdBu_r)
    #set name of color labels in hover popup
    f.update_traces(hovertemplate=f"ATI: %{{z:.2f}}")
    f.update_layout(title=f'ATI prediction for {sp}', height=1000)
    f.update_xaxes(visible=False)
    f.update_yaxes(visible=False)
    # f.update_layout(plot_bgcolor='rgba(193,222,247,1)')
    f.update_layout(plot_bgcolor='rgba(72,168,149,1)')
    return f
