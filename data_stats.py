import pandas as pd
from pandas.api.types import (
    is_object_dtype, is_categorical_dtype, is_string_dtype,
    is_bool_dtype, is_numeric_dtype,
)

class DataStats:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def number_of_columns(self):
        return self.df.shape[1]

    def number_of_rows(self):
        return self.df.shape[0]

    def column_classify(self):
        column_types_dict = {'Categorical': 0, 'Numeric': 0, 'Boolean': 0}
        columns_list = list(self.df.columns)

        for column in columns_list:
            if is_bool_dtype(self.df[column]):
                column_types_dict['Boolean'] += 1
            elif (
                is_object_dtype(self.df[column]) or
                is_string_dtype(self.df[column]) or
                is_categorical_dtype(self.df[column])
            ):
                column_types_dict['Categorical'] += 1
            elif is_numeric_dtype(self.df[column]):
                column_types_dict['Numeric'] += 1

        return column_types_dict
