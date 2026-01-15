from core.discovery import Problem, ProblemType, SchemaChecks
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.sql import functions as psf
from pyspark.sql import DataFrame


class PreProcessVariables:
    def __init__(self, dataframe: DataFrame, problem: Problem, schema_checks: SchemaChecks):
        self.dataframe = dataframe
        self.problem = problem
        self.schema_checks = schema_checks


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
        categorical_columns.remove(self.problem.target) #removing target column from the categlorical variable list

        all_columns = self.dataframe.columns
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
        preprocessing_model = pipeline.fit(self.dataframe)

        transformed_df = preprocessing_model.transform(self.dataframe)

        feature_names = []
        for i, _col in enumerate(categorical_columns):
            for label in preprocessing_model.stages[i].labels:
                feature_names.append(_col+"_"+label)

        print(feature_names)

        # Feature Mapping
        feature_map = {}
        feature_idx = 0
        
        # Categorical features
        for col in feature_names:
            # Find StringIndexer
            for stage in preprocessing_model.stages:
                if isinstance(stage, StringIndexer) and stage.getInputCol() == col:
                    categories = stage.labels
                    # OneHotEncoder creates n-1 features
                    for i in range(len(categories) - 1):
                        feature_map[f"f{feature_idx}"] = f"{col}_{categories[i]}"
                        feature_idx += 1
                    break
        
        # Numerical features
        for col in all_columns:
            feature_map[f"f{feature_idx}"] = col
            feature_idx += 1


        return transformed_df, feature_names, feature_output_col, feature_map