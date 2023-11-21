import random
import numpy as np
import pandas as pd

r_sources_path = "../data/facebook/r_sources.csv"

result = [random.sample(range(4039), 5) for _ in range(100)]
print(result)

pd.DataFrame(result).to_csv(r_sources_path, header=True, index=False)

print(random.choice(result))

data = np.asarray(pd.read_csv(r_sources_path)).tolist()

print(data)

