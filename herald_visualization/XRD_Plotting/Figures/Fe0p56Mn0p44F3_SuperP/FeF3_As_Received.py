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

csv_file_1 = r'C:\Users\rjhar\Desktop\Practice\Figures\Fe0p56Mn0p44F3_SuperP\Fe0p56Mn0p44F3_60to25wt_SuperP_100min_20-70deg.csv'
csv_file_2 = r'C:\Users\rjhar\Desktop\Practice\Figures\Fe0p56Mn0p44F3_SuperP\241203-Fe0p56Mn0p44F3-ball_milled_24h-20-70deg-40min.csv'
MnF2 = r'C:\Users\rjhar\Desktop\Practice\Figures\Fe0p56Mn0p44F3_SuperP\MnF2.csv'

# Read each CSV file separately into a DataFrame
df1 = pd.read_csv(csv_file_1)
df2 = pd.read_csv(csv_file_2)

df6 = pd.read_csv(MnF2)


# df7 = pd.read_csv(LiFeF3)


# Scale the 'Intensity' values for df1 (normalize between 0 and 1)
df1['Scaled_Intensity'] = df1['Intensity']+2000
df2['Scaled_Intensity'] = df2['Intensity']




# Initialize the plot
plt.figure(figsize=(10, 6))

# Plot each DataFrame separately on the same graph
plt.plot(df2['2Theta'], df2['Scaled_Intensity'], linewidth=1, color='black', label=r'$Fe_{0.56}Mn_{0.44}F_3$')
plt.plot(df1['2Theta'], df1['Scaled_Intensity'], linewidth=1, color='blue', label=r'$Fe_{0.56}Mn_{0.44}F_3 + SuperP$')




# Set x-range (for example, from 10 to 70)
plt.xlim(20, 70)
plt.ylim(0, 15000)

# Add labels and title
plt.xlabel('2Theta')  # Customize the label
plt.ylabel('Intensity')  # Customize the label
plt.yticks([])
plt.tick_params(axis='x', direction='out', length=10)

for pos, intensity in zip(df6['2Theta (degrees)'], df6['Intensity']):
    plt.vlines(pos, 0, 20*intensity, color='black', linewidth=1)

plt.plot([20, 70],[2500, 2500],color='black', linewidth= 1)


#for pos, intensity in zip(df6['2Theta (degrees)'], df6['Intensity']):
#    plt.vlines(pos, 2500, 20*intensity+2500, color='black', linewidth=1.5)

#plt.plot([20, 70],[5000, 5000],color='black', linewidth= 1)



# Add a legend
plt.legend(loc='upper right', reverse=True)

# Show the plot
plt.show()
