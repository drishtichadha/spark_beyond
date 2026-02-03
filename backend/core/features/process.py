from backend.core.discovery import Problem, ProblemType, SchemaChecks
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.sql import functions as psf
from pyspark.sql import DataFrame


class PreProcessVariables:
    def __init__(self, dataframe: DataFrame, problem: Problem, schema_checks: SchemaChecks, train_dataframe: DataFrame = None):
        self.dataframe = dataframe
        self.problem = problem
        self.schema_checks = schema_checks
        self.train_dataframe = train_dataframe


    def target_processing(self):
        """Process target table by desired value."""
        if self.problem.type == ProblemType.classification:
            return self.dataframe.withColumn(
                self.problem.target,
                psf.when(
                    psf.col(self.problem.target) == psf.lit(self.problem.desired_result), 
                    1
                ).otherwise(0) # This catches both non-matches AND nulls
            )
        else:
            return self.dataframe

    def categorical_processing_stages(self):
        """Index categorical variables and convert them into one hot encoded values"""

        categorical_columns = self.schema_checks.get_typed_col(col_type="categorical")
        # categorical_columns.remove(self.problem.target) #removing target column from the categlorical variable list
        if self.problem.target in categorical_columns:
            categorical_columns.remove(self.problem.target)

        all_columns = self.dataframe.columns
        if self.problem.target in all_columns:
            all_columns.remove(self.problem.target)

        all_columns = [_val for _val in all_columns if _val not in categorical_columns]

        feature_output_col = "features"

        stages = []

        stages += [StringIndexer(inputCol=_col, outputCol=_col+"_index") for _col in categorical_columns]
        stages += [OneHotEncoder(inputCol=_col+"_index", outputCol=_col+"_encoded") for _col in categorical_columns]

        assembler = VectorAssembler(inputCols=[_col+"_encoded" for _col in categorical_columns] + all_columns, outputCol=feature_output_col)
        featuer_input_col = categorical_columns + all_columns
        stages.append(assembler)
        return stages, categorical_columns, feature_output_col, all_columns
    

    def process(self):
        """All pre-processing of data"""
        self.dataframe = self.target_processing()

        stages, categorical_columns, feature_output_col, all_columns = self.categorical_processing_stages()

        pipeline = Pipeline(stages=stages)
        if self.train_dataframe is not None:
            preprocessing_model = pipeline.fit(self.train_dataframe)
        else:
            preprocessing_model = pipeline.fit(self.dataframe)

        transformed_df = preprocessing_model.transform(self.dataframe)

        feature_names = []
        for i, _col in enumerate(categorical_columns):
            for label in preprocessing_model.stages[i].labels:
                feature_names.append(_col+"_"+label)

        print(feature_names)

        # Feature Mapping - map feature indices to descriptive names
        # The VectorAssembler combines features in order:
        # 1. One-hot encoded categorical features (each category except last becomes a feature)
        # 2. Numerical features (in order of all_columns)
        feature_map = {}
        feature_idx = 0

        # Categorical features - one-hot encoding creates features for each category value
        # feature_names already contains proper names like "job_admin.", "job_blue-collar", etc.
        for feature_name in feature_names:
            feature_map[f"f{feature_idx}"] = feature_name
            feature_idx += 1

        # Numerical features - add each numerical column
        for col in all_columns:
            feature_map[f"f{feature_idx}"] = col
            feature_idx += 1

        # Also create reverse mapping (name to index) for convenience
        feature_name_to_idx = {v: k for k, v in feature_map.items()}

        return transformed_df, feature_names, feature_output_col, feature_map