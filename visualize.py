import numpy as np
import matplotlib.pyplot as plt

def visualize_data_histogram(angles, num_bins = 23):
    # print a histogram to see which steering angle ranges are most overrepresented
    avg_samples_per_bin = len(angles) / num_bins
    hist, bins = np.histogram(angles, num_bins)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width)
    plt.plot((np.min(angles), np.max(angles)), (avg_samples_per_bin, avg_samples_per_bin), 'k-')
    plt.show()

def visualize_p_data_histogram(df, num_bins = 23):
    # print a histogram to see which steering angle ranges are most overrepresented
    plt.figure()
    df.plot.hist(stacked=True, bins=num_bins, by=df['angle'])#, column="angle")
    plt.show()
