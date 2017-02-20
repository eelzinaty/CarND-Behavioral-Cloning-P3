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


def display_images(images, labels, cmap=None, num_rows=3):
    """
        Displays an image array a grid with number of rows=num_rows
    """
    print(labels)
    num_images = len(images)
    num_cols = int(num_images / num_rows)
    fig, axes = plt.subplots(num_rows, num_cols)

    for i, ax in enumerate(axes.flat):
        # Only plot the valid images
        if i < num_images:
            img = images[i]

            # Plot image.
            if (cmap):
                ax.imshow(img, cmap=cmap)
            else:
                ax.imshow(img)
            #ax.title(labels[i]+"");
    plt.show()