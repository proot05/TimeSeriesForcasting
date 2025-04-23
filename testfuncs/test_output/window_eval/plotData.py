import os
import shutil
import re
import pandas as pd
import matplotlib.pyplot as plt

# Create (or recreate) the plot directory
plot_dir = 'plot'
if os.path.exists(plot_dir):
    shutil.rmtree(plot_dir)
os.makedirs(plot_dir)

# Load the Excel file without header
df_raw = pd.read_excel('Data.xlsx', sheet_name=0, header=None)

# Use row 3 (index 2) as header names
headers = df_raw.iloc[2]
df_raw.columns = headers

# Columns to plot (B–E)
y_columns = list(headers[1:5])

for col_name in y_columns:
    fig, ax = plt.subplots()

    # Plot New LSTM Validation (rows 8–16)
    x1 = df_raw.iloc[7:16]['Future Prediction Time (s)']
    y1 = df_raw.iloc[7:16][col_name]
    ax.plot(x1, y1, linewidth=3, marker='o', color='tab:blue', label='New LSTM Validation')

    # Plot Old LSTM Validation (rows 25–33)
    x2 = df_raw.iloc[24:33]['Future Prediction Time (s)']
    y2 = df_raw.iloc[24:33][col_name]
    ax.plot(x2, y2, linewidth=3, marker='o', color='tab:orange', label='Old LSTM Validation')

    # Plot New LSTM Train dot (row 4)
    x4, y4 = df_raw.iloc[3]['Future Prediction Time (s)'], df_raw.iloc[3][col_name]
    ax.plot(x4, y4, marker='x', markersize=15, color='tab:green', linestyle='None', label='New LSTM Train')

    # Plot Old LSTM Train dot (row 21)
    x21, y21 = df_raw.iloc[20]['Future Prediction Time (s)'], df_raw.iloc[20][col_name]
    ax.plot(x21, y21, marker='x', markersize=15, color='tab:red', linestyle='None', label='Old LSTM Train')

    ax.set_xlabel('Future Prediction Time (s)')
    ax.set_ylabel(col_name)
    ax.set_title(f'{col_name} vs Future Prediction Time (s)')
    ax.legend()

    # Save the figure
    safe_name = re.sub(r'[^A-Za-z0-9_.-]+', '_', col_name)
    filepath = os.path.join(plot_dir, f"{safe_name}.png")
    fig.savefig(filepath, bbox_inches='tight')
    plt.close(fig)
