import json

path = r'c:/Users/vishe/OneDrive/Desktop/SUL_PROJECT/IT_Notebook_TCS.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

target_id = 'e4c494aa'
cell = next(c for c in nb['cells'] if c.get('id') == target_id)
assert cell['cell_type'] == 'code'

new_code = """# pip install gdeltdoc   <- run once in your .venv terminal if not installed
from gdeltdoc import GdeltDoc, Filters
import pandas as pd
import numpy as np

gdelt_start = '2018-01-01'
gdelt_end   = '2025-04-27'

# Multi-word keyword without inner quotes - gdeltdoc URL-encodes correctly.
# Country filter dropped; it is unsupported on older gdeltdoc and we will
# filter Indian-origin sources downstream if needed.
filters = Filters(
    keyword='Tata Consultancy Services',
    start_date=gdelt_start,
    end_date=gdelt_end,
)

gd = GdeltDoc()

try:
    tone_df = gd.timeline_search('timelinetone', filters)
    print(f"Tone timeline rows: {len(tone_df)}")
    print(tone_df.head())
except Exception as e:
    print(f"Tone fetch failed: {type(e).__name__}: {e}")
    tone_df = pd.DataFrame(columns=['datetime', 'GDELT_Tone'])

try:
    volume_df = gd.timeline_search('timelinevolraw', filters)
    print(f"\\nVolume timeline rows: {len(volume_df)}")
    print(volume_df.head())
except Exception as e:
    print(f"Volume fetch failed: {type(e).__name__}: {e}")
    volume_df = pd.DataFrame(columns=['datetime', 'GDELT_Volume'])"""

cell['source'] = [line + '\n' for line in new_code.split('\n')[:-1]] + [new_code.split('\n')[-1]]
cell['execution_count'] = None
cell['outputs'] = []

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
    f.write('\n')

print(f'Patched cell {target_id}')
