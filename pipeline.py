import pandas as pd
from pandas.api.types import (
    is_object_dtype, is_categorical_dtype, is_string_dtype,
    is_bool_dtype, is_numeric_dtype, is_datetime64_any_dtype
)

class Preprocessing:
    def __init__(self, data):
        self.data = data

    def read_csv(self):
        if isinstance(self.data, str):
            self.data = pd.read_csv(self.data)

    def duplicate(self):
        self.data.drop_duplicates(inplace=True)

    def missing_values(self):
        missing_columns = []
        dic = dict(self.data.isna().sum())
        for key, value in dic.items():
            if value != 0:
                missing_columns.append(key)

        for column in missing_columns:
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
