import csv
from mp_api.client import MPRester
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

with MPRester(api_key="JC3w6wXNHNGpOZBp09GoI7XuXSsII0sN") as mpr:
    # first retrieve the relevant structure
    structure = mpr.get_structure_by_material_id("mp-754407")

# important to use the conventional structure to ensure
# that peaks are labelled with the conventional Miller indices
sga = SpacegroupAnalyzer(structure)
conventional_structure = sga.get_conventional_standard_structure()

# this example shows how to obtain an XRD diffraction pattern
# these patterns are calculated on-the-fly from the structure
calculator = XRDCalculator(wavelength="CuKa")
pattern = calculator.get_pattern(conventional_structure)

# Extract the 2theta and intensity values
two_theta = pattern.x
intensities = pattern.y

# Write the data to a CSV file
csv_filename = 'Li2FeS2.csv'
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["2Theta (degrees)", "Intensity"])  # Header row
    
    # Write each 2theta and intensity pair to the CSV file
    for theta, intensity in zip(two_theta, intensities):
        writer.writerow([theta, intensity])

print(f"XRD pattern has been saved to {csv_filename}")