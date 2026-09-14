import csv
import numpy as np
import imageio.v3 as iio

'''
Small helpers for generating synthetic greyscale video / beep score fixtures,
used by the automated tests so they don't depend on external media files.
'''

def writeSyntheticGreyscaleVideo(path, gridRows=10, gridCols=10, numFrames=8, fps=10):
    # A bright square sweeps diagonally across an otherwise dim frame, giving
    # every visual column some contrast to respond to over the clip.
    frames = []
    for t in range(numFrames):
        frame = np.full((gridRows, gridCols), 40, dtype=np.uint8)
        center = int(t * (min(gridRows, gridCols) - 1) / max(numFrames - 1, 1))
        rowStart, rowEnd = max(0, center - 1), min(gridRows, center + 2)
        colStart, colEnd = max(0, center - 1), min(gridCols, center + 2)
        frame[rowStart:rowEnd, colStart:colEnd] = 220
        frames.append(frame)
    iio.imwrite(path, frames, fps=fps, macro_block_size=1)
    return path

def writeSyntheticBeepScore(path, events=None):
    # events: list of (time_ms, frequency_hz, amplitude) tuples
    if events is None:
        events = [(0, 440.0, 1.0), (25, 1800.0, 1.0), (50, 3200.0, 0.8)]
    with open(path, "w", newline="") as beepFile:
        writer = csv.writer(beepFile)
        writer.writerow(["time_ms", "frequency_hz", "amplitude"])
        for event in events:
            writer.writerow(event)
    return path
