import pandas as pd

SELECT_BY = "f1"
GREATER_IS_BETTER = SELECT_BY not in ("logloss",)

print(GREATER_IS_BETTER)