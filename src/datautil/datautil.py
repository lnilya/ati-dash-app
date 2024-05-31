from typing import List, Optional

import pandas as pd

import paths as PATHS


def getSpeciesInfo(mergeWith:pd.DataFrame = None, excludeSpeciesWithNoInfo:bool = True, useCols:List=None):
    """Retrieves the species info from the NVS species names file. If desired will also append it to any other dataframe that has
    species names in it. This is useful for example when you want to add the species info to the occurrence data.

    :param mergeWith: If not None, will merge the species info with this dataframe. The species names column must be called "PreferredSpeciesName"
    :param excludeSpeciesWithNoInfo: If True, will exclude species that are not in the species info dataframe.
    :param useCols: If not None, will only load these columns from the species info dataframe.
    :return: mergeWith + speciesInfo dataframes (if mergeWith != NONE) or just speciesInfo
    """

    if useCols is not None and "Species" not in useCols:
        useCols = useCols + ["Species"]
    speciesInfo = pd.read_csv(PATHS.SpeciesInfo.Full, usecols=useCols)

    if mergeWith is not None:

        if excludeSpeciesWithNoInfo:
            excluded = mergeWith[~mergeWith.Species.isin(speciesInfo.Species.unique())].Species.unique()
            ns = len(excluded)
            if ns > 0:
                print(f"Excluding {ns} species with no info from the species info dataframe.",excluded)

            mergeWith = mergeWith[mergeWith.Species.isin(speciesInfo.Species.unique())]

        mergeWith = mergeWith.merge(speciesInfo, how="left", left_on="Species", right_on="Species")

        return mergeWith,speciesInfo

    return speciesInfo


def getAllPlotInfo(preserveObservationIDs:bool = True, geoProperties:Optional[List] = None, mergeWith:pd.DataFrame = None):
    """Returns a dataframe with all plot information
    :param preserveObservationIDs: If True, the dataframe will contain the original observation IDs. This means some parentIDs will have multiple entries.
    If False will also remove the Year column
    :param geoProperties: Loads the properties file and adds the static properties from NZENVDS layers.
    """
    if geoProperties is None:
        plotCoords = pd.read_csv(PATHS.PlotInfo.WithGeoProps)
    else:
        plotCoords = pd.read_csv(PATHS.PlotInfo.WithGeoProps, usecols=["ParentPlotID","ParentPlotObsID"] + geoProperties)


    if not preserveObservationIDs:
        if "Year" in plotCoords.columns:
            plotCoords.drop(columns=["Year"], inplace=True)
        plotCoords = plotCoords.groupby("ParentPlotID").first().reset_index()
        plotCoords.drop(columns=["ParentPlotObsID"], inplace=True)

    if mergeWith is not None:
        mergeCols = ["ParentPlotID"]
        if preserveObservationIDs and "ParentPlotObsID" in mergeWith.columns:
            mergeCols.append("ParentPlotObsID")
        mergeWith = mergeWith.merge(plotCoords, how="left", on=mergeCols)

        return mergeWith,plotCoords

    return plotCoords