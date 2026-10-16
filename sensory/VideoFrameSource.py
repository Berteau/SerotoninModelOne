import numpy as np
import imageio.v3 as iio

class VideoFrameSource:
    """
    Decodes a greyscale video file into a sequence of retinotopic firing-rate
    frames, one per video frame, each downsampled to a (gridRows x gridCols) grid.
    """

    def __init__(self, path, gridRows=10, gridCols=10, minRateHz=1.0, maxRateHz=40.0):
        self.path = path
        self.gridRows = gridRows
        self.gridCols = gridCols
        self.minRateHz = minRateHz
        self.maxRateHz = maxRateHz
        self.rateFrames = self._loadRateFrames()

    def _loadRateFrames(self):
        rateFrames = []
        for frame in iio.imiter(self.path):
            grey = self._toGreyscale(frame)
            downsampled = self._downsample(grey)
            rateFrames.append(self._toRate(downsampled))
        if len(rateFrames) == 0:
            raise ValueError("Video source %s contained no frames" % self.path)
        return rateFrames

    @staticmethod
    def _toGreyscale(frame):
        frame = np.asarray(frame, dtype=float)
        if frame.ndim == 3:
            frame = frame[..., :3].mean(axis=-1)
        return frame

    def _downsample(self, frame):
        # Average-pool the source frame down to (gridRows x gridCols), using bin
        # edges so this works for any source resolution, not just exact multiples.
        rowEdges = np.linspace(0, frame.shape[0], self.gridRows + 1).astype(int)
        colEdges = np.linspace(0, frame.shape[1], self.gridCols + 1).astype(int)
        out = np.zeros((self.gridRows, self.gridCols))
        for r in range(self.gridRows):
            rowStart, rowEnd = rowEdges[r], max(rowEdges[r + 1], rowEdges[r] + 1)
            for c in range(self.gridCols):
                colStart, colEnd = colEdges[c], max(colEdges[c + 1], colEdges[c] + 1)
                out[r, c] = frame[rowStart:rowEnd, colStart:colEnd].mean()
        return out

    def _toRate(self, greyGrid):
        normalized = greyGrid / 255.0
        return self.minRateHz + normalized * (self.maxRateHz - self.minRateHz)

    def frameCount(self):
        return len(self.rateFrames)

    def rateFrameAt(self, frameIndex):
        return self.rateFrames[frameIndex % len(self.rateFrames)]

    def rateFrameAtTime(self, timeMs, frameDurationMs):
        frameIndex = int(timeMs // frameDurationMs)
        return self.rateFrameAt(frameIndex)
