import pandas as pd
import re

# Load and parse the log file
data = []
pattern = re.compile(r'Method:\s*(\w+),.*?k:\s*(\d+),.*?Control Ratio:\s*(\d+)%\s*Accuracy:\s*([\d.]+)%?,\s*Execution time:\s*([\d.]+)')
with open('logAnalysis/log.txt') as f:
    for line in f:
        m = pattern.search(line)
        if m:
            method, k, ratio, acc, time = m.groups()
            data.append({
                'Method': method,
                'k': k,
                'Control Ratio': int(ratio),
                'Accuracy': float(acc),
                'Execution Time': float(time)
            })

df = pd.DataFrame(data)

# Compute summary statistics for each Method & Control Ratio combination

print(df.groupby(['k'])['Accuracy'].describe())