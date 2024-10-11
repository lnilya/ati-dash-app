import enum
from typing import List, Dict, Union
import re
import pandas as pd
import plotly.express as px
class PlotProps:
    DistanceToTheCoast = "DistanceToTheCoast"
    Latitude = "Latitude"
    Longitude = "Longitude"
    DistanceToNearestRiver = "DistanceToNearestRiver"
    DistanceToTheRoad = "DistanceToTheRoad"
    Elevation = "Elevation"
    MeanAnnualHumidity = "MeanAnnualHumidity"
    BIO12 = "BIO12"
    BIO19 = "BIO19"
    BIO14 = "BIO14"
    BIO17 = "BIO17"
    PrecipitationSeasonalityRatio = "PrecipitationSeasonalityRatio"
    BIO15 = "BIO15"
    BIO18 = "BIO18"
    BIO13 = "BIO13"
    BIO16 = "BIO16"
    Slope = "Slope"
    SoilAcidity = "SoilAcidity"
    SoilAge = "SoilAge"
    CalciumContent = "CalciumContent"
    SoilChemLims = "SoilChemLims"
    Drainage = "Drainage"
    Induration = "Induration"
    pH = "pH"
    ParticleSize = "ParticleSize"
    SolarRadiationAnnualPotential = "SolarRadiationAnnualPotential"
    MeanAnnualSolarRadiation = "MeanAnnualSolarRadiation"
    SolarRadiationWinter = "SolarRadiationWinter"
    SolarRadiationWinterPotential = "SolarRadiationWinterPotential"
    SunHRSRatio = "SunHRSRatio"
    AnnualTemperatureAmplitude = "AnnualTemperatureAmplitude"
    BIO7 = "BIO7"
    GrowingDegreeDays16cBase = "GrowingDegreeDays16cBase"
    GrowingDegreeDays5cBase = "GrowingDegreeDays5cBase"
    BIO3 = "BIO3"
    BIO5 = "BIO5"
    BIO1 = "BIO1"
    BIO11 = "BIO11"
    BIO2 = "BIO2"
    BIO9 = "BIO9"
    BIO10 = "BIO10"
    BIO8 = "BIO8"
    BIO6 = "BIO6"
    MinimumTemperatureOfJulyWinter = "MinimumTemperatureOfJulyWinter"
    NormalisedMinimumWinterTemperature = "NormalisedMinimumWinterTemperature"
    BIO4 = "BIO4"
    FlowDir = "FlowDir"
    GeoMorphons = "GeoMorphons"
    NormHeight = "NormHeight"
    Position = "Position"
    Roughness = "Roughness"
    Ruggedness = "Ruggedness"
    ValleyDepth = "ValleyDepth"
    Wetness = "Wetness"
    WindExposition = "WindExposition"
    MeanAnnualVapourPressureDeficit = "MeanAnnualVapourPressureDeficit"
    OctoberVapourPressureDeficit = "OctoberVapourPressureDeficit"
    FaoPenman = "FaoPenman"
    FaoPenmanPet = "FaoPenmanPet"
    ForestPenman = "ForestPenman"
    PriestleyTaylor = "PriestleyTaylor"
    RainfallToPotentialEvapotranspirationRatio = "RainfallToPotentialEvapotranspirationRatio"
    AnnualWaterDficit = "AnnualWaterDficit"
    MeanAnnualWindspeed = "MeanAnnualWindspeed"
    EcoSystem = "EcoSystem"
    MediumSalinity = "MediumSalinity"
    MediumSoilCarbon = "MediumSoilCarbon"

    ##Inherited Props
    Northing = "Northing"
    Easting = "Easting"

    Categories = {
        DistanceToTheCoast:"Topography",
        Latitude:"Location",
        Longitude:"Location",
        DistanceToNearestRiver:"Topography",
        Elevation: "Topography",
        MeanAnnualHumidity:"Evapotranspiration",
        BIO12:"Precipitation",
        BIO19:"Precipitation",
        BIO14:"Precipitation",
        BIO17:"Precipitation",
        PrecipitationSeasonalityRatio:"Precipitation",
        BIO15:"Precipitation",
        BIO18:"Precipitation",
        BIO13:"Precipitation",
        BIO16:"Precipitation",
        Slope:"Terrain",
        SoilAcidity:"Soil",
        SoilAge:"Soil",
        CalciumContent:"Soil",
        SoilChemLims:"Soil",
        Drainage:"Soil",
        Induration:"Soil",
        pH:"Soil",
        ParticleSize:"Soil",
        SolarRadiationAnnualPotential:"Solar Radiation",
        MeanAnnualSolarRadiation:"Solar Radiation",
        SolarRadiationWinter:"Solar Radiation",
        SolarRadiationWinterPotential:"Solar Radiation",
        SunHRSRatio:"Solar Radiation",
        AnnualTemperatureAmplitude:"Temperature",
        BIO7:"Temperature",
        GrowingDegreeDays16cBase:"Temperature",
        GrowingDegreeDays5cBase:"Temperature",
        BIO3:"Temperature",
        BIO5:"Temperature",
        BIO1:"Temperature",
        BIO11:"Temperature",
        BIO2:"Temperature",
        BIO9:"Temperature",
        BIO10:"Temperature",
        BIO8:"Temperature",
        BIO6:"Temperature",
        MinimumTemperatureOfJulyWinter:"Temperature",
        NormalisedMinimumWinterTemperature:"Temperature",
        BIO4:"Temperature",
        FlowDir:"Topography",
        GeoMorphons:"Topography",
        NormHeight:"Topography",
        Position:"Terrain",
        Roughness:"Terrain",
        Ruggedness:"Terrain",
        ValleyDepth:"Terrain",
        Wetness:"Soil",
        WindExposition:"Wind",
        MeanAnnualVapourPressureDeficit:"Evapotranspiration",
        OctoberVapourPressureDeficit:"Evapotranspiration",
        FaoPenman:"Evapotranspiration",
        FaoPenmanPet:"Evapotranspiration",
        ForestPenman:"Evapotranspiration",
        PriestleyTaylor:"Evapotranspiration",
        RainfallToPotentialEvapotranspirationRatio:"Evapotranspiration",
        AnnualWaterDficit:"Precipitation",
        MeanAnnualWindspeed:"Wind",
        EcoSystem:"Classification",
        MediumSalinity:"Soil",
        MediumSoilCarbon:"Soil",
        Northing: "Location",
        Easting: "Location",
    }

    CategoryColors = {
        'Temperature': "red",
        'Precipitation': "blue",
        'Wind': "green",
        'Location': "purple",
        'Classification': "orange",
        'Evapotranspiration': "yellow",
        'Terrain': "pink",
        'Soil': "brown",
        'Topography': "cyan",
        'Fluctuation': "gray",
        'Solar Radiation': "magenta",
        'Other': "#90ee90"
    }
    colors = {
        'BIOEnd1':[ '#FF0000',  'Temperature',"Annual Mean Temperature"  ],# Bright Red
        'BIOEnd3':[ '#FF6666',  'Temperature',"Isothermality"], #Light Red
        'BIOEnd7':[ '#CC0000',  'Temperature',"Temperature Annual Range"], #Dark Red
        'BIOEnd12':[ '#215A92',  'Precipitation',"Annual Precipitation"], #Dodger Blue
        'BIOEnd15':[ '#1E90FF','Precipitation',"Precipitation Seasonality" ],  #Light Blue

        'DistanceToNearestRiver': ['#40E0D0','Hydrology'],  # Turquoise

        'Aspect Eastness':[ '#CD853F','Topography and Aspect'],  # Peru
        'Aspect Northness': ['#BC8F8F','Topography and Aspect' ], #Rosy Brown
        'Elevation': ['#8B4513','Topography and Aspect'],  # SaddleBrown
        'Slope':[ '#A0522D','Topography and Aspect'],  # Sienna

        'Latitude':[ '#808080','Geographic'],  # Gray

        'pH': ['#228B22',  'Soil'],# Forest Green

        'WindExposition':[ '#FFD700','Climate'],  # Gold

        'RedDeer':[ '#800080','Wildlife'],  # Purple
    }


    BioClimNamesAndUnits= {
        "BIO1": ["Annual Mean Temperature", "°C","Temperature"],
        "BIO2": ["Mean Diurnal Range ","°C","Fluctuation"],
        "BIO3": ["Isothermality", "%","Fluctuation"],
        "BIO4": ["Temperature Seasonality", "%","Fluctuation"],
        "BIO5": ["Max Temperature of Warmest Month", "°C","Temperature"],
        "BIO6": ["Min Temperature of Coldest Month", "°C","Temperature"],
        "BIO7": ["Temperature Annual Range", "°C","Fluctuation"],
        "BIO8": ["Mean Temperature of Wettest Quarter", "°C","Temperature"],
        "BIO9": ["Mean Temperature of Driest Quarter", "°C","Temperature"],
        "BIO10": ["Mean Temperature of Warmest Quarter", "°C","Temperature"],
        "BIO11": ["Mean Temperature of Coldest Quarter", "°C","Temperature"],
        "BIO12": ["Annual Precipitation", "mm","Precipitation"],
        "BIO13": ["Precipitation of Wettest Month", "mm","Precipitation"],
        "BIO14": ["Precipitation of Driest Month", "mm","Precipitation"],
        "BIO15": ["Precipitation Seasonality", "%","Fluctuation"],
        "BIO16": ["Precipitation of Wettest Quarter", "mm","Precipitation"],
        "BIO17": ["Precipitation of Driest Quarter", "mm","Precipitation"],
        "BIO18": ["Precipitation of Warmest Quarter", "mm","Precipitation"],
        "BIO19": ["Precipitation of Coldest Quarter", "mm","Precipitation"],
    }
    BioClimNamesAndUnitsShort = {
        "BIO1": ["MAT", "°C","Temperature"],
        "BIO2": ["Mean Diurnal Range","°C","Fluctuation"],
        "BIO3": ["Isothermality", "%","Fluctuation"],
        "BIO4": ["Temp Seasonality", "%","Fluctuation"],
        "BIO5": ["Max Temp of Warmest Mo", "°C","Temperature"],
        "BIO6": ["Min Temp of Coldest Mo", "°C","Temperature"],
        "BIO7": ["Temp Annual Range", "°C","Fluctuation"],
        "BIO8": ["Mean Temp Wettest Qu", "°C","Temperature"],
        "BIO9": ["Mean Temp Driest Qu", "°C","Temperature"],
        "BIO10": ["Mean Temp Warmest Qu", "°C","Temperature"],
        "BIO11": ["Mean Temp Coldest Qu", "°C","Temperature"],
        "BIO12": ["AP", "mm","Precipitation"],
        "BIO13": ["P Wettest Mo", "mm","Precipitation"],
        "BIO14": ["P Driest Mo", "mm","Precipitation"],
        "BIO15": ["P Seasonality", "%","Fluctuation"],
        "BIO16": ["P Wettest Qu", "mm","Precipitation"],
        "BIO17": ["P Driest Qu", "mm","Precipitation"],
        "BIO18": ["P Warmest Qu", "mm","Precipitation"],
        "BIO19": ["P Coldest Qu", "mm","Precipitation"],
    }

    @classmethod
    def renameClearTextToBioClim(cls, data:list,useShortNames = False):
        mapping = cls.BioClimNamesAndUnitsShort if useShortNames else cls.BioClimNamesAndUnits
        #reverse mapping
        mapping = {v[0]:k for k,v in mapping.items()}

        return [mapping.get(d,d) for d in data]


    @classmethod
    def renameBioClimToClearTextList(cls, data:Union[str,List[str]], returnCategories:bool = False, strFormatByOcc = {}, useShortNames:bool = False):

        mapping = cls.BioClimNamesAndUnitsShort if useShortNames else cls.BioClimNamesAndUnits

        regex = re.compile(r'.*?BIO.*?(\d+)')
        isString = isinstance(data,str)
        if isString: data = [data]

        cats = ["Other"] * len(data)
        for i,d in enumerate(data):
            repl = regex.sub(r'BIO\1', d)
            if repl in mapping and "PC_" not in d:
                d = mapping[repl][0]
                cats[i] = mapping[repl][2]
                for k,v in strFormatByOcc.items():
                    if k in data[i]:
                        d = v%d
                        break

                data[i] = d

        if isString:
            data = data[0]
            cats = cats[0]

        if returnCategories:
            return data, cats
        return data

    @classmethod
    def renameBioClimToClearText(cls, df:pd.DataFrame, useCol:str = None, useShortNames:bool = False):
        """Renames the columns of the dataframe from e.g. rBIONorm12 to clear text "Annual Precipiation".
            If useCol is not None, will use that column to rename the values instead.
            """

        regex = re.compile(r'.*?BIO.*?(\d+)')

        mapping = cls.BioClimNamesAndUnitsShort if useShortNames else cls.BioClimNamesAndUnits

        if useCol is not None:
            def repl(x):
                repl = regex.sub(r'BIO\1', x)
                if repl in mapping:
                    repl = mapping[repl][0]
                return repl
            df[useCol] = df[useCol].apply(repl)
        else:
            for c in df.columns:
                #use regex to match any BIO i
                #Use regex to match any *BIO*(number) and transform to clear text
                repl = regex.sub(r'BIO\1',c)
                if repl in mapping:
                    repl = mapping[repl][0]

                df.rename(columns={c:repl},inplace=True)

        return df

    @classmethod
    def filterOutInherited(cls, props:List):
        return [p for p in props if p not in [cls.Northing, cls.Easting]]
    @classmethod
    def allProperties(cls):
        return [x for x in dir(cls) if not x.startswith("__") and not x == "Categories" and not x == "BioClimNamesAndUnits" and not callable(getattr(cls,x))]
    @classmethod
    def categoriesToProperties(cls,cats:List):
        return [x for x in cls.allProperties() if cls.Categories[x] in cats]
    @classmethod
    def getAllCategories(cls):
        return set(cls.Categories.values())

    @classmethod
    def getBioClimDict(cls):
        return {k:v[0] for k,v in cls.BioClimNamesAndUnits.items()}

