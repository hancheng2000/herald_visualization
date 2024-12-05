import xmltodict
import pandas as pd
import numpy as np

def xrdml2csv(xrdml_file, csv_file):
    with open(xrdml_file) as f:
        xrdml = xmltodict.parse(f.read())
    axis_dict = xrdml['xrdMeasurements']['xrdMeasurement']['scan']['dataPoints']['positions'][0]
    assert axis_dict['@axis'] == '2Theta', 'Only 2Theta axis is supported'
    startPos = float(axis_dict['startPosition'])
    endPos = float(axis_dict['endPosition'])

    data = xrdml['xrdMeasurements']['xrdMeasurement']['scan']['dataPoints']['intensities']['#text']
    intensities = np.array(data.split(), dtype=float)
    two_theta = np.linspace(startPos, endPos, len(intensities))

    df = pd.DataFrame({'2Theta': two_theta, 'Intensity': intensities})
    df.to_csv(csv_file, index=False)
    return df