import pandas as pd
import re
from collections import Counter

# Read the data
df = pd.read_json('data/news_data_text_30000.jsonl', lines=True)

# Get sentences containing "Quelle:"
quelle_sentences = []
quelle_patterns = []

for idx, row in df.iterrows():
    text = row.get('text', row.get('content', ''))
    
    if 'Quelle:' in str(text):
        quelle_sentences.append({
            'id': row.get('id', idx),
            'text': text
        })
        
        # Extract everything after "Quelle:" until end of sentence or newline
        matches = re.findall(r'Quelle:([^\n.!?]*)', str(text))
        quelle_patterns.extend(matches)

print(f"Found {len(quelle_sentences)} sentences containing 'Quelle:'\n")

# Count pattern occurrences
pattern_counts = Counter(quelle_patterns)

# Filter patterns that occur more than once
recurring_patterns = {pattern: count for pattern, count in pattern_counts.items() if count > 1}

# Sort by frequency
sorted_patterns = sorted(recurring_patterns.items(), key=lambda x: x[1], reverse=True)

print(f"Found {len(recurring_patterns)} patterns that occur more than once:\n")
print("="*80)

for pattern, count in sorted_patterns[:20]:  # Show top 20
    print(f"Count: {count:4d} | Pattern: 'Quelle:{pattern}'")

print("="*80)

# Save results to CSV
results = []
for pattern, count in sorted_patterns:
    results.append({
        'pattern': f'Quelle:{pattern}',
        'count': count
    })

if results:
    results_df = pd.DataFrame(results)
    results_df.to_csv('quelle_patterns.csv', index=False)
    print(f"\nAll {len(results)} recurring patterns saved to 'quelle_patterns.csv'")

# Also save sentences with Quelle for inspection
sentences_df = pd.DataFrame(quelle_sentences)
sentences_df.to_csv('quelle_sentences.csv', index=False)
print(f"All {len(quelle_sentences)} sentences with 'Quelle:' saved to 'quelle_sentences.csv'")