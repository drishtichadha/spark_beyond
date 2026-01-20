from pyspark.sql import DataFrame

def process_col_names(dataframe: DataFrame):
        return dataframe.toDF(*[c.replace(".", "_") for c in dataframe.columns])