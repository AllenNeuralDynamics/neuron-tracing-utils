import argparse
import os

import pandas as pd
from ast import literal_eval
import matplotlib.pyplot as plt
from pandas import DataFrame


def read_csv_data(file_path):
    """
    Reads and processes data from a CSV file.

    Args:
    - file_path (str): Path to the CSV file.

    Returns:
    - DataFrame: Processed data.
    """
    df = pd.read_csv(file_path, header=None, names=["coordinate", "error"], delim_whitespace=True)
    df['coordinate'] = df['coordinate'].apply(lambda x: literal_eval(x.strip(',')))
    df['x'] = df['coordinate'].apply(lambda x: x[0])
    df['y'] = df['coordinate'].apply(lambda x: x[1])
    df['z'] = df['coordinate'].apply(lambda x: x[2])
    df.drop(columns=['coordinate'], inplace=True)

    return df


def error_counts(df):
    """
    Counts the number of errors in each category.

    Args:
    - df (DataFrame): Processed data.

    Returns:
    - dict: A dictionary containing the total number of errors and the number of errors in each category.
    """
    total_errors = df.shape[0]
    error_counts = df['error'].value_counts().to_dict()

    return {"Total Errors": total_errors, "Error Counts": error_counts}


def plot_error_distribution(df):
    """
    Visualizes the distribution of errors in a 3D scatter plot.

    Args:
    - df (DataFrame)
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    colors = {'split': 'red', 'merge': 'green', 'cross': 'blue', 'omit': 'yellow'}

    for error_type, color in colors.items():
        subset = df[df['error'] == error_type]
        ax.scatter(subset['x'], subset['y'], subset['z'], c=color, label=error_type, s=50, alpha=0.6)

    ax.set_xlim([df['x'].min(), df['x'].max()])
    ax.set_ylim([df['y'].min(), df['y'].max()])
    ax.set_zlim([df['z'].min(), df['z'].max()])
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Z Coordinate')
    ax.set_title('Distribution of Error Types')
    ax.legend()

    plt.show()


def save_errors_as_swc(df: DataFrame, out_dir):
    """
    Saves the coordinates of the errors as an SWC file.
    Each error type gets one SWC file where all the coordinates
    of that type are root nodes. The header of each swc contains
    a line with a color specification where the
    rgb values are floating point on [0, 1]. Each error type should have a unique
    contrasting color for easy visualization in Horta.

    Args:
    - df (DataFrame)
    - file_path (str): Path to the SWC file.
    """
    color_map = {
        'split': '1.0,0.0,0.0',
        'merge': '0.0,1.0,0.0',
        'cross': '0.0,0.0,1.0',
        'omit': '1.0,1.0,0.0',
    }
    for error_type in df['error'].unique():
        subset = df[df['error'] == error_type]
        with open(os.path.join(out_dir, f'{error_type}.swc'), "w") as f:
            f.write(f"# COLOR {color_map[error_type]}\n")
            for idx, row in enumerate(subset.itertuples(index=False)):
                f.write(f"{idx + 1} 0 {row.x} {row.y} {row.z} 2 -1\n")


def main(args):
    df = read_csv_data(args.input)

    counts = error_counts(df)
    print("Total Number of Errors:", counts["Total Errors"])
    print("\nNumber of Errors in Each Category:")
    for error_type, count in counts["Error Counts"].items():
        print(f"{error_type.capitalize()}: {count} errors")

    plot_error_distribution(df)

    save_errors_as_swc(df, os.path.dirname(args.input))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=str, help="path to the CSV file containing the errors",
        default=r"C:\Users\cameron.arshadi\Documents\seg-errors-653980.csv"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
