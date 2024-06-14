from typing import Literal, Optional, Union

from GlobalParams import GlobalParams
from PathBase import *
from src.classes.Enums import ClassCombinationMethod, ClassificationProblem, ModelType, NoiseRemovalBalance
from src.classes.VariableList import VariableList

"""
OCCURRENCE DATA - Contains presence of species in plots.
"""

plotFolder: str = REPOFOLDER + "_plots/"
plotFolderFigures: str = REPOFOLDER + "_plots/_figures/"

class Cache:
    SetSimilaritySpeciesCounts = REPOFOLDER + "_plots/SetSimilaritySpeciesCounts.csv"
    SetSimilarity:str = REPOFOLDER + "_plots/SetSimilarityCounts.csv"
    SetSimilarityRemData:str = REPOFOLDER + "_plots/SetSimilarityRem.csv"

    BufferFolder:str = REPOFOLDER + "_databuffer/"

    @staticmethod
    def toPickle(filename:str)->str:
        return REPOFOLDER + '_plots/%s.pickle'%filename
class Raw:
    # Raw occurrence data
    # _subfolder = "/NVS Data Aug 23"
    _subfolder = ""

    RecceData: str = DATAFOLDER + "_v2/_RawData" + _subfolder + "/Recce.csv"
    SaplingData = DATAFOLDER + '_v2/_RawData' + _subfolder + '/Saplings.csv'
    StemData = DATAFOLDER + '_v2/_RawData' + _subfolder + '/Stems.csv'
    SeedlingData = DATAFOLDER + '_v2/_RawData' + _subfolder + '/Seedlings.csv'

    # Raw Plot Info data
    SiteDescription = DATAFOLDER + '_v2/_RawData' + _subfolder + '/_SiteDescription_WithTP.csv'
    OrongoCoordinates = DATAFOLDER + '_v2/_RawData' + _subfolder + '/ORONGORONGO Plot Coordinates.csv'

    # Species Info
    SpeciesNames = DATAFOLDER + "_v2/_RawData/CurrentNVSNames_Raw.csv"

    # 2 - GIS DATA FOR PLOT INFO
    GisEcosystems = GISFOLDER + 'lris BasicEcosystems/'
    GisSoilLayers = GISFOLDER + 'Iris-SoilLayers/'
    GisNZEnvDSFolder = GISFOLDER + 'NZEnvDS_v1-1/'
    GisHotRunzFolder = GISFOLDER + 'HotRUNZ/'
    GisGRazingFolder = GISFOLDER + 'Grazing/'

    # 3 - CLIMATE DATA FOR PLOTINFO
    ClimHotRunzRain = GISFOLDER + 'HotRUNZ/nni-rain/'
    ClimHotRunzTMax = GISFOLDER + 'HotRUNZ/nni-tmax/'
    ClimHotRunzTMin = GISFOLDER + 'HotRUNZ/nni-tmin/'

class GISExport:
    TrainingDataSubset = GISFOLDER + "EftiExport/TrainingDataSubset.csv"

class Occ:
    # Separate occurrence sets for each data colleciton method
    Recce: str = DATAFOLDER + "_v3/_TrainingData/1_Occurence_Recce.csv"
    Stem: str = DATAFOLDER + "_v3/_TrainingData/1_Occurence_Stems.csv"
    Seedling: str = DATAFOLDER + "_v3/_TrainingData/1_Occurence_Seedlings.csv"
    Sapling: str = DATAFOLDER + "_v3/_TrainingData/1_Occurence_Saplings.csv"

    # Merged full dataset
    Combined: str = DATAFOLDER + "_v3/_TrainingData/2_Occurence_Full.csv"


class PlotInfo:
    # PLOT INFO - Contains things like coordinates, dates etc.  

    # Full plot info
    NoProps: str = DATAFOLDER + "_v3/_TrainingData/1_PlotInfo_Full.csv"
    WithGeoProps: str = DATAFOLDER + "_v3/_TrainingData/2_PlotInfo_WithProps.csv"
    SameYearRemeasurements: str = DATAFOLDER + "_v3/_TrainingData/2_SameYearRemeasuredPlots.csv"


class SpeciesInfo:
    Full = DATAFOLDER + "rawdata/1_SpeciesInfo.csv"


class Bioclim:
    AtPlotsByYear = DATAFOLDER + '_v3/_TrainingData/1_BioClimAllYearsAndPlots.csv'
    AtPlotsByYearLinearAppx = DATAFOLDER + '_v3/_TrainingData/2_ClimateEstimates_HotRunzLinear.csv'  # linear approximation values from horunz


class Maps:
    AdultsOnly = "assets/data/maps/"
    AdultsWithSameSplitByDBH = "assets/data/maps_dbh/"
class Shifts:

    allPlotsByMethod = DATAFOLDER + f'_v3/_TrainingData/1_AllMigrations_byMethod.csv'
    @staticmethod
    def allPlotsCombined(by: Union[str, ClassCombinationMethod], mergedAR:bool = True) -> str:
        if isinstance(by, ClassCombinationMethod):
            by = by.value

        if not mergedAR:
            return DATAFOLDER + f'_v3/_TrainingData/2_AllMigrations_{by}_ARID.csv'

        return DATAFOLDER + f'_v3/_TrainingData/2_AllMigrations_{by}.csv'
    @staticmethod
    def allPlotsCombinedPCInfo(by: ClassCombinationMethod, mergedAR:bool = True) -> str:
        if not mergedAR:
            return DATAFOLDER + f'_v3/_TrainingData/2_AllMigrations_{by.value}_ARID_PCAInfo.pickle'

        return DATAFOLDER + f'_v3/_TrainingData/2_AllMigrations_{by.value}_PCAInfo.pickle'

    @staticmethod
    def trainingPointDifficultyAll(by: ClassCombinationMethod, cp:ClassificationProblem, reps = 10) -> str:
        return DATAFOLDER + f'_v3/_TrainingData/3_PredictionDifficulty_{by.value}_{cp.value}_{reps}.csv'
    @staticmethod
    def trainingPointDifficulty(by: ClassCombinationMethod, vars:VariableList, cp:ClassificationProblem, reps = 10) -> str:
        return DATAFOLDER + f'_v3/_TrainingData/3_PredictionDifficulty_{by.value}_{vars.name}_{cp.value}_{reps}.csv'
    @staticmethod
    def trainingPointDifficultySub(by: ClassCombinationMethod, vars:VariableList, cp:ClassificationProblem, model:ModelType, reps = 10) -> str:
        return DATAFOLDER + f'_v3/_TrainingData/3_PredictionDifficulty_{by.value}_{vars.name}_{cp.value}_{reps}_{model.value}.csv'

    @staticmethod
    def allPlotsPCALoadings(by: Literal["Species", "Genus"]) -> str:
        return DATAFOLDER + f'_v3/_TrainingData/1_AllMigrationsPCALoadings_by_{by}.csv'


class Noise:

    possibleNoiseNetwork = DATAFOLDER + '_v3/_Analysis/NoiseNetwork.csv'

    @staticmethod
    def noisePerformance(comb:ClassCombinationMethod, vars:VariableList, problem:ClassificationProblem, balance:NoiseRemovalBalance)->str:
        return  DATAFOLDER + '_v3/_Models/NoisePerformance_%s_%s_%s%s.csv'%(comb.value,vars.name,problem.value,f"_{balance.value}" if balance != NoiseRemovalBalance.Combined else "")
    @staticmethod
    def noiseLabels(comb:ClassCombinationMethod, vars:VariableList, problem:ClassificationProblem)->str:
        return  DATAFOLDER + '_v3/_Models/NoiseScores_%s_%s_%s.csv'%(comb.value,vars.name,problem.value)

class VarSelection:
    varSelectionBySpecies = DATAFOLDER + '_v3/_Analysis/1_VariableSelection_bySpecies_mrmr_scores_%s.csv'
    varSelectionTotal = DATAFOLDER + '_v3/_Analysis/1_VariableSelection_Total_mrmr_scores_%s.csv'

class Results:
    stackedFeaturesFolder = DATAFOLDER + '_v3/_Analysis/StackedFeatures/'
    similaritiesFolder = DATAFOLDER + 'Similarities/'
    predictionsMeanFolder = DATAFOLDER + 'MeanPredictions/'
    predictionsFolder = DATAFOLDER + '_v3/_Analysis/YearPredictions/'
    predictionImgsFolder = DATAFOLDER + '_v3/_Analysis/PredictionImgs/'
    permutationImportanceFolder = DATAFOLDER + '_v3/_Analysis/PermutationImportance/'
    permutationImportance = DATAFOLDER + '_v3/_Analysis/PermutationImportance/PermutationImportance.csv'
    gradientAnalysis = DATAFOLDER + '_v3/_Analysis/EFTIGradientAnalysis.csv'


    @staticmethod
    def correlation(comb:ClassCombinationMethod):
        s = DATAFOLDER + 'drivers/Correlation_%s.csv'
        return s%comb.value
    @staticmethod
    def fi(comb:ClassCombinationMethod):
        s = DATAFOLDER + 'drivers/FI_%s.csv'
        return s%comb.value


class TrainedModels:

    Scores = DATAFOLDER + '_v3/_Models/_Scores.csv'
    Results = DATAFOLDER + '_v3/_Models/_Results.csv'
    ModelFolder = DATAFOLDER + '_v3/_Models/Classifiers/'

    @staticmethod
    def byDiffScores(model:ModelType, label:str):
        return DATAFOLDER + f'_v3/_Models/_Scores_{model.value}_{label}.csv'
    @staticmethod
    def resultsByMVP(model:ModelType, var:VariableList, comb:ClassCombinationMethod, nr:float, nrb:NoiseRemovalBalance):
        return DATAFOLDER + f'_v3/_Models/_Results_{model.value}_{var.name}_{comb.value}_{nr:.2f}_{nrb.value}.csv'

    @staticmethod
    def model(modelID:Optional[str], subfolder:str = None)->str:
        if subfolder is None:
            return DATAFOLDER + '_v3/_Models/Classifiers/%s.pickle'%modelID
        else:
            if isinstance(subfolder,float) and subfolder == 0:
                subfolder = "0.0"

            #if no model is provided will just give the subfolder this model would be saved in.
            if modelID is None:
                return DATAFOLDER + '_v3/_Models/Classifiers/%s/'%(subfolder)
            return DATAFOLDER + '_v3/_Models/Classifiers/%s/%s.pickle'%(subfolder,modelID)
