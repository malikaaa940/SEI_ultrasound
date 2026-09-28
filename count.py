import os
import pandas as pd

def load_dataset(folder):
    records = []
    for fname in sorted(os.listdir(folder)):
        if not fname.lower().endswith('.jpg'):
            continue
        name_upper = fname.upper()
        if name_upper.startswith('OPTIMAL'):
            label = 'optimal'
        elif name_upper.startswith('SUBOPTIMAL'):
            label = 'suboptimal'
        elif name_upper.startswith('UNLABELED'):
            label = 'unlabelled'
        else:
            label = 'unknown'
        records.append({'filename': fname,
                        'path': os.path.join(folder, fname),
                        'label': label})
    return pd.DataFrame(records)

df = load_dataset('/Users/malikaeryomina/SEI_AI_ultrasound/Ultrasound_AI_Project/dataset/image_grayscale')
print(df['label'].value_counts())
# Optimal       → training data (label = 0)
# Suboptimal    → training data (label = 1)
# Unlabelled    → inference only (no label needed)