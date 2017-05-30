import sys
import numpy as np
import pandas as pd
from utils import DataWriter

predictions = np.zeros(shape=(416,))
file_cnt = 0

for filename in sys.argv[1:-1]:
    print('read from file: {}'.format(filename))
    file_cnt += 1

    df = pd.read_csv(filename)
    cases = df['total_cases'].values
    predictions = predictions + cases

# ensemble with average
predictions = np.around(predictions / file_cnt).astype('int64')

# replace with new total_cases
df['total_cases'] = predictions

df.to_csv(sys.argv[-1], index=False)
