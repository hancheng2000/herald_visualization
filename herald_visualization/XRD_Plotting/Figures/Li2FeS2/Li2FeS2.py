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

csv_file_2 = r'C:\Users\rjhar\Desktop\Practice\Figures\Li2FeS2\250213_Li2FeS2_100min_20-70deg_no_background.xy'
Li2S = r'C:\Users\rjhar\Desktop\Practice\Figures\Li2FeS2\Li2S.csv'
Li2FeS2 = r'C:\Users\rjhar\Desktop\Practice\Figures\Li2FeS2\Li2FeS2.csv'

# Read each CSV file separately into a DataFrame

df2 = pd.read_csv(csv_file_2, delim_whitespace=True)
df3 = pd.read_csv(Li2S)
df4 = pd.read_csv(Li2FeS2)

# df7 = pd.read_csv(LiFeF3)


# Scale the 'Intensity' values for df1 (normalize between 0 and 1)

df2['Scaled_Intensity'] = df2['Intensity']+3000



# Initialize the plot
plt.figure(figsize=(10, 6))

# Plot each DataFrame separately on the same graph
plt.plot(df2['2Theta'], df2['Scaled_Intensity'], color='black', label=r'$Fe + S + Li_2S$')





# Set x-range (for example, from 10 to 70)
plt.xlim(20, 70)
plt.ylim(0, 12000)

# Add labels and title
plt.xlabel('2Theta')  # Customize the label
plt.ylabel('Intensity')  # Customize the label
plt.yticks([])
plt.tick_params(axis='x', direction='out', length=10)

for pos, intensity in zip(df4['2Theta (degrees)'], df4['Intensity']):
    plt.vlines(pos, 0, 10*intensity, color='black', linewidth=1)

plt.plot([20, 70],[1500, 1500],color='black', linewidth= 1)

for pos, intensity in zip(df3['2Theta (degrees)'], df3['Intensity']):
    plt.vlines(pos, 1500, 1500+10*intensity, color='black', linewidth=1)

plt.plot([20, 70],[3000, 3000],color='black', linewidth= 1)


#for pos, intensity in zip(df6['2Theta (degrees)'], df6['Intensity']):
#    plt.vlines(pos, 2500, 20*intensity+2500, color='black', linewidth=1.5)

#plt.plot([20, 70],[5000, 5000],color='black', linewidth= 1)



# Add a legend
plt.legend(loc='upper right', reverse=True)

# Show the plot
plt.show()
