import pandas as pd
import matplotlib.pyplot as plt
import io

# 1. Load the data
# (Using the data provided earlier)
df = pd.read_csv("benchmark_results16x16.csv")

# 2. Pre-process the labels for the X-axis
# Extract the numeric resolution (e.g., '16' from 'speed_16x16')
df['res'] = df['test_type'].str.extract('(\d+)').astype(int)

# 3. Calculate means and standard deviation for error bars
summary = df.groupby(['res', 'processor'])['duration_ms'].agg(['mean', 'std']).unstack()

# 4. Create the Visualization
plt.figure(figsize=(10, 6))

# Plot CPU
plt.errorbar(summary.index, summary['mean']['cpu'], yerr=summary['std']['cpu'], 
             label='CPU', marker='o', color='#d62728', linewidth=2, capsize=5)

# Plot GPU
plt.errorbar(summary.index, summary['mean']['gpu'], yerr=summary['std']['gpu'], 
             label='GPU', marker='s', color='#1f77b4', linewidth=2, capsize=5)

# 5. Styling
plt.title("Chunk Generation Scaling: CPU vs. GPU Performance", fontsize=14, fontweight='bold')
plt.xlabel("Resolution (N x N)", fontsize=12)
plt.ylabel("Mean Duration (ms)", fontsize=12)
plt.xticks(summary.index) # Ensure x-axis shows 16, 32, 64, 128
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(title="Processor", fontsize=11)

# Annotate the final gap
gap = summary['mean']['cpu'].iloc[-1] / summary['mean']['gpu'].iloc[-1]
plt.annotate(f'{gap:.1f}x Speedup', 
             xy=(128, summary['mean']['gpu'].iloc[-1]), 
             xytext=(100, 500),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8))

plt.tight_layout()
plt.savefig("scaling_comparison.png", dpi=300)
plt.show()

print("Chart generated: scaling_comparison.png")