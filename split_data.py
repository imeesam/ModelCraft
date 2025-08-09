from pipeline import Preprocessing
from sklearn.model_selection import train_test_split

target_column_user = "target"  


pp = Preprocessing("data.csv")
pp.read_csv()
pp.duplicate()
pp.missing_values()
pp.fix_dtypes()
pp.encoding()
pp.scaling()

top_features = pp.corr(target_column=target_column_user)

if target_column_user not in pp.data.columns:
    raise ValueError(f"Target column '{target_column_user}' not found in data.")

X = pp.data[top_features]
y = pp.data[target_column_user]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

print("Top Features:", top_features)
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
