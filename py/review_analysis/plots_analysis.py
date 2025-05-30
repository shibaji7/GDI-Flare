import matplotlib.pyplot as plt
import scienceplots
plt.style.use(["science", "ieee"])
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Tahoma", "DejaVu Sans", "Lucida Grande", "Verdana"]
plt.rcParams.update(
    {
        "text.usetex": False,
    }
)

import pandas as pd

def plot_histograms(
        df:pd.DataFrame, column:str, 
        bins:int=500, title:str="Density: Velocity", 
        xlabel:str="Velocity, m/s", ylabel:str="Density/Frequency",
        filename:str="histogram.png",
        abs_value:bool=False, color="red", ls="-", lw=0.4
    ):
    fig = plt.figure(figsize=(3, 3), dpi=300)
    ax = fig.add_subplot(111)
    ax.hist(
        df[column].abs() if abs_value else df[column], bins=bins, 
        density=False, color=color, ls=ls, lw=lw,
        alpha=0.7, histtype="step"
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_yscale("log")
    ax.grid()
    plt.tight_layout()
    fig.savefig(filename, dpi=300)
    return