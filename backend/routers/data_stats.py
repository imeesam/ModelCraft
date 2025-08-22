import pandas as pd
from pandas.api.types import (
    is_object_dtype, is_categorical_dtype, is_string_dtype,
    is_bool_dtype, is_numeric_dtype,is_datetime64_any_dtype
)

class DataStats:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def number_of_columns(self):
        return self.df.shape[1]

    def number_of_rows(self):
        return self.df.shape[0]

    def column_classify(self):
        column_types_dict = {'Categorical': 0, 'Numeric': 0, 'Boolean': 0,'DateTime':0}
        columns_list = list(self.df.columns)

        for column in columns_list:
            if is_bool_dtype(self.df[column]):
                column_types_dict['Boolean'] += 1
            elif (is_datetime64_any_dtype(self.df[column])):
                column_types_dict['DateTime']+=1
            elif (
                is_object_dtype(self.df[column]) or
                is_string_dtype(self.df[column]) or
                is_categorical_dtype(self.df[column])
            ):
                column_types_dict['Categorical'] += 1
            elif is_numeric_dtype(self.df[column]):
                column_types_dict['Numeric'] += 1

        return column_types_dict
    def missing_values(self):
        total = self.df.shape[0] * self.df.shape[1]
        missing_count = self.df.isna().sum().sum()
        missing_percentage = (missing_count / total) * 100

        return {
            "missing_count": missing_count,
            "missing_percentage": round(missing_percentage, 2)
    }
    def duplicates(self):
        total_rows = self.df.shape[0]
        duplicate_count = self.df.duplicated().sum()
        duplicate_percentage = (duplicate_count / total_rows) * 100 if total_rows > 0 else 0

        return {
            'duplicate_count': duplicate_count,
            'duplicate_percentage': round(duplicate_percentage, 2)
        }
    def data_size(self):
        memory_bytes = self.df.memory_usage(deep=True).sum()
        memory_mb = memory_bytes / (1024**2)   
        return round(memory_mb, 2)  
    
    def column_names(self):
        columns_list=list(self.df.columns)
        return columns_list
    def numerical_summary(self):
        summary_list = []
        numeric_cols = self.df.select_dtypes(include='number').columns
        
        for col in numeric_cols:
            summary_list.append({
                "Column": col,
                "Mean": round(self.df[col].mean(), 2),
                "Median": round(self.df[col].median(), 2),
                "Std Dev": round(self.df[col].std(), 2),
                "Min": round(self.df[col].min(), 2),
                "Max": round(self.df[col].max(), 2),
                "Missing %": round((self.df[col].isna().sum() / len(self.df)) * 100, 2)
            })
        
        return pd.DataFrame(summary_list)







