import re
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# filepaths
deepseek_file = "problem_ratings_42_math_deepseek (seed).txt"
openai_file = "problem_ratings_42_math_openai (seed).txt"

def extract_ratings(filepath):
    """Parses the text file and extracts all the integer ratings."""
    ratings = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            # Find all instances of "Rating: [number]"
            matches = re.findall(r'Rating:\s*(\d+)', content)
            ratings = [int(m) for m in matches]
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
    return ratings

# data
ds_ratings = extract_ratings(deepseek_file)
oa_ratings = extract_ratings(openai_file)

print(f"Loaded {len(ds_ratings)} DeepSeek ratings and {len(oa_ratings)} OpenAI ratings.")

# count frequencies of each rating (1 through 10), initializing w 0 for x axis
ds_counts = Counter({i: 0 for i in range(1, 11)})
oa_counts = Counter({i: 0 for i in range(1, 11)})

ds_counts.update(ds_ratings)
oa_counts.update(oa_ratings)

# in order from 1 to 10
labels = np.arange(1, 11)
ds_values = [ds_counts[i] for i in labels]
oa_values = [oa_counts[i] for i in labels]

# plot
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(labels)) 
width = 0.35  

# bars
rects1 = ax.bar(x - width/2, ds_values, width, label='DeepSeek-V3', color='#2ca02c', edgecolor='black', linewidth=0.7)
rects2 = ax.bar(x + width/2, oa_values, width, label='OpenAI GPT-4o', color='#1f77b4', edgecolor='black', linewidth=0.7)

# text and all
ax.set_ylabel('Frequency (Number of Problems)', fontsize=12, weight='bold')
ax.set_xlabel('Interestingness Score (1-10)', fontsize=12, weight='bold')
ax.set_title('Baseline Rating Distributions: DeepSeek vs. OpenAI', fontsize=14, weight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11)
ax.legend(fontsize=11)
#wanted to remove spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# save as png
output_filename = 'score_distributions.png'
plt.tight_layout()
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f"Successfully generated and saved '{output_filename}'")

# plt.show() # uncomment if you want it to pop up on your screen