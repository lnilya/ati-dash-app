from typing import Dict, Optional

import numpy as np

from GlobalParams import GlobalParams
from src.classes.Enums import ClassificationProblem, ClassCombinationMethod
from src.classes.Serializable import Serializable
from src.classes.VariableList import VariableList


class ClassifierDataSet(Serializable):
    """Stores training or test data a classifier can be using"""

    @staticmethod
    def fromDict(dict:Optional[Dict]) -> Optional['ClassifierDataSet']:
        if dict is None: return None
        return ClassifierDataSet(
            dict["_noiseRed"],
            ClassCombinationMethod[dict["_combinationMethod"]],
            ClassificationProblem[dict["_classificationProblem"]],
            VariableList.fromDict(dict["vars"]),
            dict["X"],
            dict["y"],
            dict.get("plotIDs", None),
            dict.get("params", {}))

    def __init__(self, noiseRed:float, comMethod: ClassCombinationMethod, classificationProblem: ClassificationProblem,
                 vars: VariableList, X: np.array, y: np.array, plotIDs: np.array = None, params: Dict = None):
        self._noiseRed = noiseRed
        self._combinationMethod = comMethod
        self._classificationProblem = classificationProblem
        self._vars = vars
        self._X = X
        self._y = y
        self._plotIDs = plotIDs
        self._params = GlobalParams().__dict__
        if params is not None: self._params.update(params)

    def toDict(self, discardData: bool = False) -> Dict:
        """Converts the object to a dictionary
           :param discardData: If true the data is not included, only the variables and the problem"""
        return {
                "_noiseRed": self._noiseRed,
            "_classificationProblem": self._classificationProblem.name,
                "_combinationMethod": self._combinationMethod.name,
                "vars": self._vars.toDict(),
                "X": self._X.astype(np.float32) if not discardData else [],
                "y": self._y if not discardData else [],
                "plotIDs": self.plotIDs if not discardData else [],
                "params": self._params
                }

    def getClassificationData(self, scale: bool = True, labelEncode: bool = True):
        """Returns the data in a format that can be used by a classifier"""
        X, y = self._X, self._y
        scaler = None
        encoder = None
        if labelEncode:
            encoder = LabelEncoder()
            y = encoder.fit_transform(y)
        if scale:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

        return X,y,scaler,encoder

    @property
    def params(self) -> Dict:
        return self._params

    @property
    def plotIDs(self) -> np.array:
        return self._plotIDs

    @property
    def numObs(self) -> int:
        return len(self._y)

    @property
    def X(self) -> np.ndarray:
        return self._X

    @property
    def y(self) -> np.ndarray:
        return self._y

    @property
    def classificationProblem(self) -> ClassificationProblem:
        return self._classificationProblem

    @property
    def combinationMethod(self) -> ClassCombinationMethod:
        return self._combinationMethod

    @property
    def noiseReduction(self) -> float:
        return self._noiseRed

    @property
    def vars(self) -> VariableList:
        return self._vars
