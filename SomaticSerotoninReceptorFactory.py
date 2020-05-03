from SomaticSerotoninReceptor import *
from DiffuseReceptorFactory import *


class SomaticSerotoninDiffuseReceptorFactory(DiffuseReceptorFactory):
    def __init__(self, typeString, weightFunction,
                 defaultWeightFunctionParamters):
        self.typeString = typeString
        self.weightFunction = weightFunction
        self.defaultWeightFunctionParamters = defaultWeightFunctionParamters

    def constructReceptor(self, initialLevel=0, weightFunctionParamters=None):
        if weightFunctionParamters is None:
            weightFunctionParamters = self.defaultWeightFunctionParamters
        return SomaticSerotoninReceptor(
            None,
            self.typeString,
            self.weightFunction(weightFunctionParamters),
            initialLevel)
