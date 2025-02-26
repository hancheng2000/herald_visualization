import pandas as pd
import matplotlib.pyplot as plt
import os

default_params = {
    'font.family': 'serif',
    'axes.labelsize': 20,
    'axes.labelweight': 'bold',  # Make axes labels bold
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'xtick.major.size': 7,
    'ytick.major.size': 7,
    'xtick.major.width': 2.0,
    'ytick.major.width': 2.0,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'font.size': 24,
    'axes.linewidth': 2.0,
    'lines.dashed_pattern': (5, 2.5),
    'lines.markersize': 10,
    'lines.linewidth': 3,
    'lines.markeredgewidth': 1,
    'lines.markeredgecolor': 'k',
    'legend.fontsize': 16,  # Adjust the font size of the legend
    'legend.title_fontsize': 24,  # Increase legend title size if needed
    'legend.frameon': True
}

plt.rcParams.update(default_params)

# CSV file paths
XRD_csv_1 = r'C:\Users\rjhar\Desktop\Practice\Figures\(Fe,Co,Ni,Cu)S2\250206_Run1_20-70deg_40min.csv'


Peaks_csv_1 = r'C:\Users\rjhar\Desktop\Practice\Figures\(Fe,Co,Ni,Cu)S2\CuS.csv'
Peaks_csv_2 = r'C:\Users\rjhar\Desktop\Practice\Figures\(Fe,Co,Ni,Cu)S2\CuS.csv'
Peaks_csv_3 = r'C:\Users\rjhar\Desktop\Practice\Figures\(Fe,Co,Ni,Cu)S2\CuS.csv'


# Read each CSV file separately into a DataFrame
df1 = pd.read_csv(XRD_csv_1)
#df2 = pd.read_csv(XRD_csv_2)
#df3 = pd.read_csv(XRD_csv_3)
df4 = pd.read_csv(Peaks_csv_1)
df5 = pd.read_csv(Peaks_csv_2)
df6 = pd.read_csv(Peaks_csv_3)


# Scale the 'Intensity' values for df1 (normalize between 0 and 1)
df1['Scaled_Intensity'] = df1['Intensity'] - 1000
#df2['Scaled_Intensity'] = df2['Intensity'] + 5000
#df3['Scaled_Intensity'] = df3['Intensity'] + 10000


# Initialize the plot
plt.figure(figsize=(10, 6))

# Plot each DataFrame separately on the same graph
plt.plot(df1['2Theta'], df1['Scaled_Intensity'],linewidth=1, label=r'$(Fe,Co,Ni,Cu)S_2$')
#plt.plot(df2['2Theta'], df2['Scaled_Intensity'], label=r'$Li(Fe_{0.75}Mn_{0.25})F_3$')
#plt.plot(df3['2Theta'], df3['Scaled_Intensity'], label=r'$LiMnF_3$')


# Set x-range (for example, from 10 to 70)
plt.xlim(20, 70)
plt.ylim(bottom=0)

# Add labels and title
plt.xlabel('2Theta')  # Customize the label
plt.ylabel('Intensity')  # Customize the label
plt.yticks([])
plt.tick_params(axis='x', direction='out', length=10)

# Add x axis peak ticks
for pos, intensity in zip(df4['2Theta (degrees)'], df4['Intensity']):
    plt.vlines(pos, 0, 3*intensity, color='black', linewidth=1.5)

plt.plot([20, 70],[500, 500],color='black', linewidth= 1)

for pos, intensity in zip(df5['2Theta (degrees)'], df5['Intensity']):
    plt.vlines(pos, 500, 500 + 3*intensity, color='black', linewidth=1.5)

plt.plot([20, 70],[1000, 1000],color='black', linewidth= 1)

for pos, intensity in zip(df6['2Theta (degrees)'], df6['Intensity']):
    plt.vlines(pos, 1000, 1000 + 3*intensity, color='black', linewidth=1.5)

plt.plot([20, 70],[1500, 1500],color='black', linewidth= 1)




# Add a legend
plt.legend(loc='upper right', reverse=True)

# Show the plot
plt.show()
