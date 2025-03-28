import os
import re
import math
import matplotlib.pyplot as plt
from collections import defaultdict

# Define the directory containing the files
directory = "slurm/mnist"

# Regex to extract fields from filenames
pattern = re.compile(r"t0_(\d+)_t1_(\d+)_(\d+)_(\w+)\.out")

# Data structure: { (limit, flag) -> { train_0 -> last_line_value } }
data = defaultdict(lambda: defaultdict(list))

# Process files
for filename in os.listdir(directory):
    if filename.endswith(".out"):
        match = pattern.match(filename)
        if match:
            train_0, train_1, limit, flag = match.groups()
            train_0, limit = int(train_0), int(limit)  # Convert to integers
            
            file_path = os.path.join(directory, filename)
            
            if os.path.isfile(file_path):
                try:
                    # Read the last line efficiently
                    last_line = None
                    with open(file_path, "r") as f:
                        for line in f:
                            last_line = line.strip()
                    
                    if last_line:  # Convert last line to float
                        last_value = float(last_line)
                        data[(limit, flag)][train_0].append(last_value)
                
                except Exception as e:
                    print(f"Skipping {filename} due to error: {e}")

# Determine grid size
num_subplots = len(data)
num_cols = math.ceil(math.sqrt(num_subplots))  # Square-like grid
num_rows = math.ceil(num_subplots / num_cols)  # Adjust for non-square cases

# Create grid of subplots
fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(4 * num_cols, 4 * num_rows))

# Flatten axes array for easier indexing (works even if num_rows * num_cols > num_subplots)
axes = axes.flatten()

# Plot data
for ax, ((limit, flag), values) in zip(axes, data.items()):
    sorted_train_0 = sorted(values.keys())  # Sort x-axis values
    y_values = [sum(values[t]) / len(values[t]) for t in sorted_train_0]  # Compute average if multiple values

    ax.plot(sorted_train_0, y_values, marker='o', linestyle='-', label=f"Samples Per Epoch {limit}, Priors= {flag}")
    ax.set_title(f"Limit {limit}, Flag {flag}")
    ax.set_xlabel("Normal Training Epochs")
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.grid(True)

# Hide unused subplots if grid is larger than needed
for i in range(len(data), len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.subplots_adjust(hspace=0.3, wspace=0.3)  # Adjust spacing between subplots

# Save the figure instead of showing it
plt.savefig("graph.png", dpi=300)
plt.close()  # Close the figure to free memory
