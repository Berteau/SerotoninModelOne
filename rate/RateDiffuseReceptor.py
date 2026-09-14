'''
Rate-model diffuse serotonin receptors, mirroring the spiking model's
SomaticSerotoninReceptor / AxonalSerotoninReceptor but deterministic.

Somatic receptors sit on a RateNeuron and contribute a diffuse current
weight * level to the neuron's total input current I. Axonal receptors sit on a
RateAxon and shift its transmission failure rate (which is rolled into the
effective weight, see RateAxon).

Sign conventions follow the *code's* existing parameters, not the methods
draft's prose (the two disagree on 5HT1A/5HT2A somatic polarity; the retinotopic
network uses the code's weights, e.g. Somatic5HT2AWeight=+20, so we stay
weight-sign-agnostic and let the passed-in weight carry the sign).
'''


class RateSomaticReceptor:
    def __init__(self, typeString, weight, initialLevel=0.0):
        self.typeString = typeString
        self.weight = weight
        self.level = initialLevel
        self.current = weight * initialLevel
        self.target = None

    def getTypeString(self):
        return self.typeString

    def setTarget(self, target):
        self.target = target

    def setLevel(self, level):
        self.level = level
        self.current = self.weight * level
        if self.target is not None:
            self.target.recomputeDiffuseCurrent()


class RateAxonalReceptor:
    def __init__(self, typeString, weight, initialLevel=0.0):
        self.typeString = typeString
        self.weight = weight
        self.level = initialLevel
        self.target = None

    def getTypeString(self):
        return self.typeString

    def setTarget(self, target):
        self.target = target
        if self.target is not None:
            self.target.recomputeFailureRate()

    def setLevel(self, level):
        self.level = level
        if self.target is not None:
            self.target.recomputeFailureRate()


class RateSomaticReceptorFactory:
    def __init__(self, typeString, weight):
        self.typeString = typeString
        self.weight = weight

    def getTypeString(self):
        return self.typeString

    def constructReceptor(self, initialLevel=0.0):
        return RateSomaticReceptor(self.typeString, self.weight, initialLevel)


class RateAxonalReceptorFactory:
    def __init__(self, typeString, weight):
        self.typeString = typeString
        self.weight = weight

    def getTypeString(self):
        return self.typeString

    def constructReceptor(self, initialLevel=0.0):
        return RateAxonalReceptor(self.typeString, self.weight, initialLevel)
