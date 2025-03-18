"""
Module containing visualization functions for data analysis,
particularly suited for biochemical data representation.
"""

from typing import List, Optional, Tuple, Union, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.axes import Axes


def plot_parallel_hists(
    counts_array: np.ndarray,
    bins_array: np.ndarray,
    labels: List[str],
    colors: List[str],
    xlabel: str,
    title: str,
    filename: str,
    lines: Optional[List[Tuple[str, float]]] = None,
    xlim: Optional[Tuple[float, float]] = None
) -> None:
    """
    Creates a plot with multiple parallel histograms.
    
    This function generates a plot displaying multiple histograms
    one above the other, allowing direct visual comparison.
    
    Parameters
    ----------
    counts_array : np.ndarray
        2D array containing histogram values.
        Each row represents a different histogram.
    bins_array : np.ndarray
        2D array containing bin boundaries for each histogram.
    labels : List[str]
        List of labels for each histogram.
    colors : List[str]
        List of colors for each histogram.
    xlabel : str
        Label for the x-axis.
    title : str
        Title of the plot.
    filename : str
        Name of the file to save the plot.
    lines : Optional[List[Tuple[str, float]]], optional
        List of tuples (name, position) to draw vertical lines.
    xlim : Optional[Tuple[float, float]], optional
        Limits (min, max) for the x-axis.
    
    Returns
    -------
    None
        The function displays and saves the plot but doesn't return any value.
    """
    # Check input dimensions
    if not len(colors) == len(labels) == np.shape(counts_array)[0]:
        print(f"Incompatible dimensions: colors={len(colors)}, labels={len(labels)}, "
              f"counts_array={np.shape(counts_array)[0]}")
        print("Dimensions do not match. The size of colors and labels must "
              "correspond to the number of histograms to be plotted.")
        return None
    
    if not np.shape(counts_array)[1] == np.shape(bins_array)[1] - 1:
        print(f"Incompatible dimensions: counts_array={np.shape(counts_array)[1]}, "
              f"bins_array={np.shape(bins_array)[1]}")
        print("Dimensions do not match. Abscissa (number of columns in "
              "bins_array) and ordinates (number of columns in counts_array) do not match.")
        return None
    
    # Create figure
    plt.figure(figsize=(20, 10))
    ax = plt.subplot(1, 1, 1)
    plt.xlabel(xlabel, fontsize=15)
    plt.yticks([], fontsize=20)
    plt.xticks(fontsize=30)
    plt.title(title, fontsize=25, pad=30)
    plt.yticks([])
    
    # Define vertical offsets for labels
    vertical_offsets = [0.7] * len(labels)  # Default value of 0.7
    # Specific adjustment for first elements if necessary
    if len(labels) > 0 and len(labels) <= 4:
        vertical_offsets[:len(labels)] = [0.65] * len(labels)
    
    # Plot histograms
    num_histograms, num_bins = np.shape(counts_array)  # Number of histograms, number of bins per histogram
    gap = np.max(counts_array) + 1/num_histograms * np.max(counts_array)
    vertical_position = 0
    
    for hist_index in range(num_histograms):
        histogram = np.zeros((num_bins, 2))
        histogram[:, 0] = bins_array[hist_index][1:]
        histogram[:, 1] = counts_array[hist_index] + vertical_position
        
        # Plot baseline
        ax.plot([np.min(bins_array), np.max(bins_array)], 
                [vertical_position, vertical_position], color='black')
        
        # Plot histogram
        ax.plot(histogram[:, 0], histogram[:, 1], linewidth=2, 
                linestyle='solid', color=colors[hist_index])
        ax.fill_between(histogram[:, 0], vertical_position, histogram[:, 1], 
                        alpha=0.3, color=colors[hist_index])
        
        # Add label
        offset_index = min(hist_index, len(vertical_offsets) - 1)
        plt.text(np.min(bins_array) + 0.1, 
                 vertical_position + vertical_offsets[offset_index] * gap, 
                 labels[hist_index], fontsize=13)
        
        vertical_position -= gap
    
    # Add vertical lines if specified
    if lines is not None:
        line_colors = ['r', 'g', 'orange', 'b', 'purple', 'c']
        for line_index, line in enumerate(lines):
            color_index = line_index % len(line_colors)
            plt.axvline(line[1], label=line[0], linestyle='--', color=line_colors[color_index])
    
    # Set x-axis limits if specified
    if xlim is not None:
        plt.xlim(xlim)
    
    # Finalize plot
    plt.legend(bbox_to_anchor=(0.01, hist_index/(2*num_histograms)), fontsize=20)
    plt.grid(linestyle='--', linewidth=0.8)
    plt.savefig(filename)
    plt.show()


def polar_density_plot(
    angles: np.ndarray,
    amplitudes: np.ndarray,
    title: str = '',
    bins: int = 100,
    cmap: str = 'twilight',
    save: bool = False,
    file: str = 'polarplot.png'
) -> None:
    """
    Creates a polar density plot to visualize pseudorotation angles.
    
    This function is particularly useful for visualizing nucleic acid
    conformations based on phase angle P and amplitude Vmax.
    
    Parameters
    ----------
    angles : np.ndarray
        Phase angles P in degrees.
    amplitudes : np.ndarray
        Corresponding Vmax amplitudes.
    title : str, optional
        Title of the plot.
    bins : int, optional
        Number of bins for the 2D histogram (currently unused).
    cmap : str, optional
        Color palette for visualization.
    save : bool, optional
        If True, saves the plot.
    file : str, optional
        Name of the file to save the plot.
    
    Returns
    -------
    None
        The function displays and saves the plot but doesn't return any value.
    
    Notes
    -----
    Edge effects may appear at the 0°-360° junction if the number of bins is small.
    """
    # Create figure
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111, polar=True)
    ax.grid(True, linewidth=1)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    
    # Add colored areas
    ax.bar(x=np.radians(18), height=60, width=np.radians(36), color="blue", alpha=0.25)  # Blue area
    ax.bar(x=np.radians(162), height=60, width=np.radians(36), color="red", alpha=0.25)  # Red area
    
    # Plot points
    ax.scatter(np.radians(angles), amplitudes)
    
    # Add conformation labels
    angle_positions = [18, 54, 90, 126, 162, 198, 234, 270, 306, 342]
    label_radius = 63  # Radial position for labels
    conformations = [
        "C3' endo", "C4' exo", "O4' endo", "C1' exo", "C2' endo",
        "C3' exo", "C4' endo", "O4' exo", "C1' endo", "C2' exo"
    ]
    
    for i, angle in enumerate(angle_positions):
        text_alignment = "left" if angle < 180 else "right"
        ax.text(x=np.radians(angle), y=label_radius, s=conformations[i],
                fontweight="bold", ha=text_alignment, fontsize=15)
    
    # Configure radial ticks
    ax.set_rmax(60)
    ax.set_rticks([0, 20, 40, 60], fontsize=15)
    ax.set_rlabel_position(108)
    ax.tick_params(axis="y", which="both", labelsize=15)
    
    # Configure angular ticks
    ax.set_xticks(np.radians(np.linspace(0, 360, 10, endpoint=False)), minor=False)
    plt.setp(ax.xaxis.get_majorticklabels()[1:5], ha="left")
    plt.setp(ax.xaxis.get_majorticklabels()[6:9], ha="right")
    
    # Remove duplicates from legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc="upper right", 
               markerscale=2, fontsize=10, bbox_to_anchor=(1.2, 1))
    
    # Finalize plot
    plt.title(title, fontsize=24)
    plt.suptitle("Pseudorotation Phase angles P(polar) and Amplitude Vmax (radius)")
    plt.tight_layout()
    
    if save:
        plt.savefig(file)
    
    plt.show()


def plot_fingerprint(
    combination: List[bool],
    title: str,
    long_names: Optional[List[str]] = None
) -> None:
    """
    Creates a fingerprint-like visualization for binary data.
    
    This function generates a heatmap that visually represents a set of
    binary data, such as molecular interactions or structural features.
    
    Parameters
    ----------
    combination : List[bool]
        Array of boolean values representing the features to visualize.
    title : str
        Title of the plot.
    long_names : Optional[List[str]], optional
        List of full names for labels. If not provided, generic labels
        will be used.
    
    Returns
    -------
    None
        The function displays the plot but doesn't return any value.
    """
    # Check that long_names is defined
    if long_names is None:
        long_names = [f"Feature_{i}" for i in range(len(combination))]
    
    # Create data for heatmap
    heatmap_data = []
    value = 4.0
    
    for feature_present in combination:
        if feature_present == False:
            heatmap_data.append([0.0, 0.0])
        else:
            heatmap_data.append([value, value])
        value += 1.0
    
    # Convert to DataFrame for easier visualization
    data_array = np.array(heatmap_data)
    data_transposed = data_array.transpose()
    feature_df = pd.DataFrame(data_transposed, columns=long_names)
    feature_df_transposed = feature_df.transpose()
    
    # Create figure
    _, ax = plt.subplots(figsize=(11, 4))
    
    # Define color palette
    cmap = sns.color_palette("cubehelix_r", as_cmap=True)
    
    # Create heatmap
    sns.heatmap(feature_df_transposed, cmap=cmap, vmin=0, vmax=18)
    
    # Finalize plot
    plt.title(title)
    plt.show()