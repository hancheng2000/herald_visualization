This package is designed for plotting the data in herald project.
The example notebook represents a typical workflow that aims to be universal for all relevant electrochemical testings in the project:
1. locate into the folder with all 4 tests
2. convert mpr to csv files
3. plot the GITT/cycling profile of different cycles
Note that the plotting functions return fig and ax in case you want further modifications to the plot (e.g. add theoretical OCV for comparison), and you need to save the figure yourself (save_png is provided as an option in function input)