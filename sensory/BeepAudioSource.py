import csv
import math
import numpy as np

class BeepAudioSource:
    """
    Loads a simple beep score - a CSV of (time_ms, frequency_hz, amplitude) rows -
    and maps it onto a 1D tonotopic array of firing rates, one band per column,
    with log-spaced center frequencies and Gaussian frequency tuning per band.
    """

    def __init__(self, path, numBands=20, minFreqHz=200.0, maxFreqHz=5000.0,
                 minRateHz=1.0, maxRateHz=40.0, tuningWidthOctaves=0.5):
        self.path = path
        self.numBands = numBands
        self.minRateHz = minRateHz
        self.maxRateHz = maxRateHz
        self.tuningWidthOctaves = tuningWidthOctaves
        self.bandCenterFreqsHz = np.logspace(math.log10(minFreqHz), math.log10(maxFreqHz), numBands)
        self.bandCenterOctaves = np.log2(self.bandCenterFreqsHz)
        self.events = self._loadEvents()

    def _loadEvents(self):
        events = []
        with open(self.path, newline="") as beepFile:
            reader = csv.DictReader(beepFile)
            for row in reader:
                events.append((float(row["time_ms"]), float(row["frequency_hz"]), float(row["amplitude"])))
        events.sort(key=lambda event: event[0])
        if len(events) == 0:
            raise ValueError("Beep source %s contained no events" % self.path)
        return events

    def stateAtTime(self, timeMs):
        # Step-hold: the most recent event at or before timeMs is in effect.
        current = self.events[0]
        for event in self.events:
            if event[0] > timeMs:
                break
            current = event
        return current[1], current[2]

    def rateVectorAtTime(self, timeMs):
        frequencyHz, amplitude = self.stateAtTime(timeMs)
        freqOctaves = math.log2(max(frequencyHz, 1e-6))
        tuning = np.exp(-((freqOctaves - self.bandCenterOctaves) ** 2) / (2 * self.tuningWidthOctaves ** 2))
        return self.minRateHz + amplitude * tuning * (self.maxRateHz - self.minRateHz)
