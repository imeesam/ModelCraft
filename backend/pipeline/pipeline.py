import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import mutual_info_classif
from pandas.api.types import (
    is_object_dtype, is_categorical_dtype, is_string_dtype,
    is_bool_dtype, is_numeric_dtype, is_datetime64_any_dtype
)

target_column=0

class Preprocessing:
    def __init__(self, data):
        self.data = data

    def read_csv(self):
        if isinstance(self.data, str):
            self.data = pd.read_csv(self.data)

    def duplicate(self):
        self.data.drop_duplicates(inplace=True)

    def missing_values(self):
        for column in self.data.columns:
            if self.data[column].isna().sum() > 0:
                if (
                    is_object_dtype(self.data[column]) or
                    is_string_dtype(self.data[column]) or
                    is_categorical_dtype(self.data[column]) or
                    is_bool_dtype(self.data[column])
                ):
                    mod = self.data[column].mode()[0]
                    self.data[column].fillna(mod, inplace=True)
                elif is_numeric_dtype(self.data[column]):
                    skew_value = self.data[column].skew()
                    if abs(skew_value) < 0.5:
                        self.data[column].fillna(self.data[column].mean(), inplace=True)
                    else:
                        self.data[column].fillna(self.data[column].median(), inplace=True)
                elif is_datetime64_any_dtype(self.data[column]):
                    median_date = self.data[column].median()
                    self.data[column].fillna(median_date, inplace=True)

    def fix_dtypes(self):
        for column in self.data.columns:
            if is_object_dtype(self.data[column]):
                non_null_values = self.data[column].dropna()
                sample = (
                    non_null_values.sample(10, random_state=42)
                    if len(non_null_values) >= 10
                    else non_null_values
                )

                try:
                    pd.to_datetime(sample, errors='raise')
                    self.data[column] = pd.to_datetime(self.data[column], errors='coerce')
                    continue
                except:
                    pass

                def is_float_string(val): 
                    try:
                        float(val)
                        return True
                    except:
                        return False

                int_like_count = 0
                float_like_count = 0

                for value in sample:
                    if isinstance(value, str):
                        if value.replace(" ", "").isdigit():
                            int_like_count += 1
                        elif is_float_string(value):
                            float_like_count += 1

                if int_like_count >= 7 and float_like_count < 7:
                    try:
                        self.data[column] = self.data[column].astype(int)
                    except Exception as e:
                        print(f"Could not convert column '{column}' to int: {e}")
                elif float_like_count >= 7:
                    try:
                        self.data[column] = self.data[column].astype(float)
                    except Exception as e:
                        print(f"Could not convert column '{column}' to float: {e}")

    def encoding(self):
        for column in list(self.data.columns):
            if (
                is_object_dtype(self.data[column]) or
                is_string_dtype(self.data[column]) or
                is_categorical_dtype(self.data[column])
            ):
                unique_values = self.data[column].nunique()

                if unique_values == 2:
                    self.data = pd.get_dummies(self.data, columns=[column], drop_first=True)
                    continue
                elif 2 < unique_values <= 6:
                    encoder = LabelEncoder()
                    self.data[column] = encoder.fit_transform(self.data[column].astype(str))
                elif unique_values > 6:
                    top_categories = self.data[column].value_counts().nlargest(5).index
                    self.data[column] = self.data[column].apply(
                        lambda x: x if x in top_categories else 'Other'
                    )
                    encoder = LabelEncoder()
                    self.data[column] = encoder.fit_transform(self.data[column].astype(str))

            elif is_bool_dtype(self.data[column]):
                self.data[column] = self.data[column].astype(int)

    def scaling(self):
        scaler = StandardScaler()
        for column in self.data.columns:
            if is_numeric_dtype(self.data[column]):
                unique_values = set(self.data[column].dropna().unique())
                if unique_values == {0, 1}:
                    continue
                try:
                    self.data[[column]] = scaler.fit_transform(self.data[[column]])
                except ValueError as e:
                    print(f"Skipping column '{column}' due to error: {e}")



    def corr(self, target_column):
        num_cols = self.data.select_dtypes(include=['number']).columns.drop(target_column, errors='ignore')
        cat_cols = self.data.select_dtypes(exclude=['number']).columns

        scores = {}

        if len(num_cols) > 0:
            correlation = self.data[num_cols].corrwith(self.data[target_column], numeric_only=True)
            scores.update(correlation.abs().to_dict())

        if len(cat_cols) > 0:
            mi_scores = mutual_info_classif(
                self.data[cat_cols],
                self.data[target_column],
                discrete_features=True
            )
            scores.update(dict(zip(cat_cols, mi_scores)))

        sorted_features = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_features = [feature for feature, _ in sorted_features if feature != target_column][:6]

        return top_features

    

