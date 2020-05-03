from DiffuseReceptor import *


class SomaticSerotoninReceptor(DiffuseReceptor):

    def setLevel(self, level):
        if isinstance(level, (int, float)):
            self.level = level
            self.current = self.weight * self.level
        else:
            raise ValueError(
                "Error: setLevel requires an integer or floating point input")

    def doActivity(self):
        self.target.diffuseCurrent += self.current
        return


# 1A offers current-based suppression (2.5‐20 nA)

#
