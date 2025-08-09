import pandas as pd
from sklearn.datasets import load_iris
from pipeline import Preprocessing  # Your preprocessing class

# Load iris dataset as a DataFrame
iris = load_iris(as_frame=True)
df = iris.frame


df["Category"] = ["A", "B", "C"] * (len(df) // 3) + ["A"] * (len(df) % 3)
df["Binary"] = [1, 0] * (len(df) // 2) + [1] * (len(df) % 2)

df.to_csv("test_data.csv", index=False)

target_column_user = "target"


pp = Preprocessing("test_data.csv")
pp.read_csv()
pp.duplicate()
pp.missing_values()
pp.fix_dtypes()
pp.encoding()
pp.scaling()

top_features = pp.corr(target_column=target_column_user)

print("Selected top features:", top_features)