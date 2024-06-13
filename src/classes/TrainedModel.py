import pickle
from typing import Dict, List, Optional

import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

from GlobalParams import GlobalParams
from src.classes.ClassifierDataSet import ClassifierDataSet
from src.classes.Enums import ModelType, ClassificationProblem, ClassCombinationMethod, NoiseRemovalBalance
from src.classes.FileIDClasses import ModelFileID
from src.classes.Serializable import Serializable
import paths as PATHS
import os


class TrainedModel(Serializable):

    """
    A class that stores all the data of a trained classifier including training and test data, scalers and label encoders.
    The idea is that this class can be pickled and stored in a file. This file can then be loaded and used to predict
    """

    modelID:ModelType

    trainingData:ClassifierDataSet
    testData:ClassifierDataSet

    scaler:StandardScaler
    labelEncoder:LabelEncoder

    testScore: float
    trainScore: float

    trainedClassifier:any

    noiseReduction:float
    noiseBalance:NoiseRemovalBalance

    _fileID: ModelFileID

    def __init__(self, species:str, modelID:ModelType|str, trainData:ClassifierDataSet, testData:ClassifierDataSet = None, scaler:StandardScaler = None, labelEncoder:LabelEncoder = None, testScore:float = -1, trainScore:float = -1, trainedClassifier:any = None, params:Dict = None, noiseReduction:float = 0, noiseBalance:NoiseRemovalBalance = NoiseRemovalBalance.Combined):
        self.species = species
        self.modelID = modelID if isinstance(modelID,ModelType) else ModelType[modelID]
        self.trainingData = trainData
        self.testData = testData
        self.scaler = scaler
        self.labelEncoder = labelEncoder
        self.testScore = testScore
        self.trainScore = trainScore
        self.trainedClassifier = trainedClassifier
        self.noiseReduction = noiseReduction if noiseReduction is not None else 0
        self.noiseBalance = noiseBalance
        self.params = GlobalParams().__dict__

        self._fileID = ModelFileID(species, self.modelID, self.trainingData.vars, self.trainingData.classificationProblem, self.trainingData.combinationMethod, self.noiseReduction, self.noiseBalance)

        if params is not None: self.params.update(params)


    def toDict(self, discardDatasets:bool = False) -> dict:
        return {
            "species": self.species,
            "modelID": self.modelID.value,
            "trainingData": self.trainingData.toDict(discardDatasets) if self.trainingData is not None else None,
            "testData": self.testData.toDict(discardDatasets) if self.testData is not None else None,
            "scaler": self.scaler,
            "labelEncoder": self.labelEncoder,
            "testScore": self.testScore,
            "trainScore": self.trainScore,
            "trainedClassifier": self.trainedClassifier,
            "noiseReduction": self.noiseReduction,
            "noiseBalance": self.noiseBalance.value,
            "params": self.params
        }

    @staticmethod
    def fromDict(dict: dict) -> "TrainedModel":
        return TrainedModel(
            dict["species"],
            ModelType[dict["modelID"]],
            ClassifierDataSet.fromDict(dict.get("trainingData",None)),
            ClassifierDataSet.fromDict(dict.get("testData",None)),
            dict["scaler"],
            dict["labelEncoder"],
            dict["testScore"],
            dict["trainScore"],
            dict["trainedClassifier"],
            dict.get("params",{}),
            dict.get("noiseReduction",0),
            NoiseRemovalBalance[dict.get("noiseBalance", "Combined").capitalize()]
        )



    def fileExists(self)->bool:
        return self._fileID.fileExists()
    def saveToDisc(self, discardDatasets:bool = False, path:str = None):
        if path is None:
            path = self._fileID.file
            self._fileID.checkAndCreateSubfolder()

        with open(path, "wb") as f:
            pickle.dump(self.toDict(discardDatasets),f)

    @staticmethod
    def load(path)-> "TrainedModel":
        """Loads the classifier. Can either be a dictionary (old way) or the class (new way). Output is always an instance of this class"""
        with open(path, "rb") as f:
            data = pickle.load(f)
            trm = TrainedModel.fromDict(data)
            trm._fileID = ModelFileID.parseFromPath(path)
            return trm