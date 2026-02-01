"""XGboost based implementation""" 
from backend.core.discovery import Problem, ProblemType
from pyspark.sql import DataFrame
import matplotlib.pyplot as plt
import pandas as pd
import json

import numpy as np

from xgboost.spark import SparkXGBRegressor, SparkXGBClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator

class FeatureSelector:
    def __init__(self,
                 problem: Problem,
                 transformed_df: DataFrame,
                 feature_names: list,
                 feature_col: str,
                 feature_idx_name_mapping: dict,
                 train_split: float = 0.8,
                 random_state: int = 42):
        self.problem = problem
        self.transformed_df = transformed_df
        self.train_split = train_split
        self.feature_col = feature_col
        self.random_state = random_state

        self.xgb_model = None
        self.train_df = None
        self.test_df = None

        self.feature_idx_name_mapping = feature_idx_name_mapping
        self.booster = None

        # Build complete feature names list from feature_idx_name_mapping
        # This includes both categorical (one-hot encoded) and numerical features
        if feature_idx_name_mapping:
            # Sort by feature index (f0, f1, f2, ...)
            sorted_features = sorted(
                feature_idx_name_mapping.items(),
                key=lambda x: int(x[0][1:])  # Extract number from 'f0', 'f1', etc.
            )
            self.feature_names = [name for _, name in sorted_features]
        else:
            self.feature_names = feature_names

    def train_test_split(self):
        train_df, test_df = self.transformed_df.randomSplit([self.train_split, 1-self.train_split], seed=42)
        return train_df, test_df
    
    def train_model(self):
        if self.problem.type == getattr(ProblemType, "classification").value:
            model = SparkXGBClassifier(
                features_col=self.feature_col,
                label_col=self.problem.target,
                prediction_col="prediction",
                num_round=100,
                max_depth=4,
                eta=0.1,
                learning_rate = 0.1,
                eval_metric="logloss",
                random_state = self.random_state
            )
        elif self.problem.type == getattr(ProblemType, "regression").value:
            model = SparkXGBRegressor(
                features_col=self.feature_col,
                label_col=self.problem.target,
                prediction_col="prediction",
                num_round=100,
                max_depth=4,
                eta=0.1,
                learning_rate = 0.1,
                eval_metric="squarederror",
                random_state = self.random_state
            )
        else:
            raise ValueError("Only classification and regression are allowed as problem types.")
        
        self.train_df, self.test_df = self.train_test_split()

        self.xgb_model = model.fit(self.train_df)
        self.booster = self.xgb_model.get_booster()


    def get_feature_importances(self):
        """
        Get feature importances with proper feature names.

        Returns:
            List of tuples (feature_name, importance_score) sorted by importance descending
        """
        feature_importances = self.xgb_model.get_feature_importances()

        importance_dict = {}
        for feature_key, importance_value in feature_importances.items():
            # feature_key is like 'f0', 'f1', etc.
            if feature_key in self.feature_idx_name_mapping:
                feature_name = self.feature_idx_name_mapping[feature_key]
            else:
                # Try to get from feature_names list by index
                try:
                    idx = int(feature_key[1:])  # Extract number from 'f0', 'f1', etc.
                    if idx < len(self.feature_names):
                        feature_name = self.feature_names[idx]
                    else:
                        feature_name = feature_key
                except (ValueError, IndexError):
                    feature_name = feature_key

            importance_dict[feature_name] = importance_value

        sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        return sorted_importance

    def plot_feature_importance(self, topn: int =None):
        sorted_importance = self.get_feature_importances()
        if topn:
            if topn <= len(sorted_importance):
                sorted_importance = sorted_importance[:topn]
            else:
                raise ValueError(f"topn should be < or = number of features: {len(sorted_importance)}")
            
        plt.figure(figsize=(10, 6))
        plt.barh([_val[0] for _val in sorted_importance], [_val[1] for _val in sorted_importance], color='steelblue')
        plt.xlabel('Importance Score')
        plt.ylabel('Features')
        plt.title('XGBoost Feature Importance - Classification')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('sparkxgb_classification_importance.png', dpi=300, bbox_inches='tight')
        print("\nFeature importance plot saved as 'sparkxgb_classification_importance.png'")

    def predict(self, train: bool = True):
        if train:
            feature_df = self.train_df
        else:
            feature_df = self.test_df

        predictions_df = self.xgb_model.transform(feature_df)

        return predictions_df
    
    def evaluate(self, train: bool = True):

        predictions_df = self.predict(train=train)

        if self.problem.type == getattr(ProblemType, "classification").value:
            # Accuracy
            evaluator_accuracy = MulticlassClassificationEvaluator(
                labelCol=self.problem.target,
                predictionCol="prediction",
                metricName="accuracy"
            )

            # F1 Score
            evaluator_f1 = MulticlassClassificationEvaluator(
                labelCol=self.problem.target,
                predictionCol="prediction",
                metricName="f1"
            )

            # Precision
            evaluator_precision = MulticlassClassificationEvaluator(
                labelCol=self.problem.target,
                predictionCol="prediction",
                metricName="weightedPrecision"
            )

            # Recall
            evaluator_recall = MulticlassClassificationEvaluator(
                labelCol=self.problem.target,
                predictionCol="prediction",
                metricName="weightedRecall"
            )

            # AUC-ROC
            evaluator_auc = BinaryClassificationEvaluator(
                labelCol=self.problem.target,
                rawPredictionCol="probability",
                metricName="areaUnderROC"
            )

            accuracy = evaluator_accuracy.evaluate(predictions_df)
            f1_score = evaluator_f1.evaluate(predictions_df)
            precision = evaluator_precision.evaluate(predictions_df)
            recall = evaluator_recall.evaluate(predictions_df)
            auc = evaluator_auc.evaluate(predictions_df)

            print(f"Accuracy:  {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall:    {recall:.4f}")
            print(f"F1 Score:  {f1_score:.4f}")
            print(f"AUC-ROC:   {auc:.4f}")

            eval_criteria = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "auc_roc": auc
            }
            return eval_criteria
        else:
            #TODO: Fill the feature selector
            return None
        
    def get_probability_impact_summary(self):
        """
        Get a summary of how each split affects classification probability.

        Returns:
            DataFrame with columns:
            - Feature: Feature name
            - Threshold: Split threshold value
            - Left_Prob: Probability when going left (value < threshold)
            - Right_Prob: Probability when going right (value >= threshold)
            - Prob_Impact: Absolute difference in probability
            - Gain: Information gain from this split
        """
        trees_json = self.booster.get_dump(dump_format='json')

        all_impacts = []

        for tree_idx, tree_json in enumerate(trees_json):
            tree = json.loads(tree_json)

            def analyze_node(node):
                if 'split' in node and 'children' in node:
                    # Get average leaf values from each branch
                    def avg_leaves(n):
                        if 'split' not in n:
                            return [n.get('leaf', 0)]
                        leaves = []
                        for child in n.get('children', []):
                            leaves.extend(avg_leaves(child))
                        return leaves

                    left_leaves = avg_leaves(node['children'][0])
                    right_leaves = avg_leaves(node['children'][1])

                    left_score = np.mean(left_leaves)
                    right_score = np.mean(right_leaves)

                    left_prob = 1 / (1 + np.exp(-left_score))
                    right_prob = 1 / (1 + np.exp(-right_score))

                    feature_idx = str(node['split'])

                    all_impacts.append({
                        'Feature': self.feature_idx_name_mapping[feature_idx] if self.feature_idx_name_mapping.get(feature_idx, None) else feature_idx,
                        'Threshold': node.get('split_condition'),
                        'Left_Prob': left_prob,
                        'Right_Prob': right_prob,
                        'Prob_Impact': abs(right_prob - left_prob),
                        'Gain': node.get('gain', 0)
                    })

                    # Recurse
                    for child in node['children']:
                        analyze_node(child)

            analyze_node(tree)

        return pd.DataFrame(all_impacts).sort_values('Prob_Impact', ascending=False)

    def get_shap_analysis(
        self,
        sample_size: int = 1000,
        plot: bool = True,
        plot_type: str = 'summary'
    ) -> dict:
        """
        Perform SHAP (SHapley Additive exPlanations) analysis to understand
        feature contributions to predictions.

        Args:
            sample_size: Number of samples to use for SHAP analysis (for performance)
            plot: Whether to generate plots
            plot_type: Type of plot - 'summary', 'bar', 'beeswarm', or 'all'

        Returns:
            Dictionary with:
            - shap_values: SHAP values array
            - feature_importance: DataFrame with mean absolute SHAP values per feature
            - sample_data: The sampled data used for analysis
            - explainer: The SHAP explainer object
        """
        try:
            import shap
        except ImportError:
            raise ImportError(
                "SHAP is not installed. Install it with: pip install shap"
            )

        if self.xgb_model is None:
            raise ValueError("Model not trained. Call train_model() first.")

        # Get the underlying XGBoost booster
        booster = self.booster

        # Sample data for SHAP analysis
        test_pdf = self.test_df.toPandas()
        if len(test_pdf) > sample_size:
            test_pdf = test_pdf.sample(n=sample_size, random_state=self.random_state)

        # Extract feature matrix from the vector column
        from pyspark.ml.linalg import DenseVector, SparseVector

        def extract_features(row):
            vec = row[self.feature_col]
            if isinstance(vec, DenseVector):
                return vec.toArray()
            elif isinstance(vec, SparseVector):
                return vec.toArray()
            else:
                return np.array(vec)

        X = np.array([extract_features(row) for _, row in test_pdf.iterrows()])

        # Create SHAP explainer
        explainer = shap.TreeExplainer(booster)
        shap_values = explainer.shap_values(X)

        # Calculate mean absolute SHAP values for feature importance
        if isinstance(shap_values, list):
            # For multi-class, use the positive class (index 1)
            shap_for_importance = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        else:
            shap_for_importance = shap_values

        mean_shap = np.abs(shap_for_importance).mean(axis=0)

        # Map to feature names using feature_idx_name_mapping
        feature_importance_shap = []
        for i, importance in enumerate(mean_shap):
            # Get proper feature name from mapping
            feature_name = self.feature_idx_name_mapping.get(f'f{i}')
            if feature_name is None:
                feature_name = self.feature_names[i] if i < len(self.feature_names) else f'feature_{i}'

            feature_importance_shap.append({
                'Feature': feature_name,
                'Mean_SHAP': importance,
                'Std_SHAP': np.std(shap_for_importance[:, i]),
                'Max_SHAP': np.max(np.abs(shap_for_importance[:, i])),
                'Min_SHAP': np.min(shap_for_importance[:, i]),
                'Max_Positive_SHAP': np.max(shap_for_importance[:, i]),
            })

        importance_df = pd.DataFrame(feature_importance_shap).sort_values(
            'Mean_SHAP', ascending=False
        )

        # Generate plots if requested
        if plot:
            # Create feature names for plotting using the mapping
            feature_names_for_plot = []
            for i in range(X.shape[1]):
                feature_name = self.feature_idx_name_mapping.get(f'f{i}')
                if feature_name is None:
                    feature_name = self.feature_names[i] if i < len(self.feature_names) else f'feature_{i}'
                feature_names_for_plot.append(feature_name)

            if plot_type in ['summary', 'all']:
                plt.figure(figsize=(12, 8))
                shap.summary_plot(
                    shap_for_importance,
                    X,
                    feature_names=feature_names_for_plot,
                    show=False
                )
                plt.title('SHAP Summary Plot')
                plt.tight_layout()
                plt.savefig('shap_summary_plot.png', dpi=300, bbox_inches='tight')
                plt.close()
                print("SHAP summary plot saved as 'shap_summary_plot.png'")

            if plot_type in ['bar', 'all']:
                plt.figure(figsize=(10, 8))
                shap.summary_plot(
                    shap_for_importance,
                    X,
                    feature_names=feature_names_for_plot,
                    plot_type='bar',
                    show=False
                )
                plt.title('SHAP Feature Importance (Bar)')
                plt.tight_layout()
                plt.savefig('shap_bar_plot.png', dpi=300, bbox_inches='tight')
                plt.close()
                print("SHAP bar plot saved as 'shap_bar_plot.png'")

        return {
            'shap_values': shap_values,
            'feature_importance': importance_df,
            'sample_data': X,
            'explainer': explainer,
            'feature_names': feature_names_for_plot if plot else self.feature_names
        }

    def get_feature_value_impacts(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get detailed analysis of how specific feature values impact predictions.
        Combines tree split analysis with aggregated statistics.

        Args:
            top_n: Number of top features to analyze

        Returns:
            DataFrame with feature value impact analysis
        """
        # Get split-level impacts
        split_impacts = self.get_probability_impact_summary()

        # Aggregate by feature
        feature_summary = split_impacts.groupby('Feature').agg({
            'Prob_Impact': ['mean', 'max', 'sum', 'count'],
            'Gain': ['mean', 'sum'],
            'Threshold': ['min', 'max', 'median']
        }).round(4)

        # Flatten column names
        feature_summary.columns = [
            'Avg_Prob_Impact', 'Max_Prob_Impact', 'Total_Prob_Impact', 'Num_Splits',
            'Avg_Gain', 'Total_Gain',
            'Min_Threshold', 'Max_Threshold', 'Median_Threshold'
        ]

        feature_summary = feature_summary.reset_index()
        feature_summary = feature_summary.sort_values('Total_Prob_Impact', ascending=False)

        return feature_summary.head(top_n)

    def explain_prediction(
        self,
        instance_idx: int = 0,
        use_test: bool = True
    ) -> dict:
        """
        Explain a single prediction using SHAP values.

        Args:
            instance_idx: Index of the instance to explain
            use_test: Whether to use test set (True) or train set (False)

        Returns:
            Dictionary with prediction explanation
        """
        try:
            import shap
        except ImportError:
            raise ImportError(
                "SHAP is not installed. Install it with: pip install shap"
            )

        if self.xgb_model is None:
            raise ValueError("Model not trained. Call train_model() first.")

        # Get data
        df = self.test_df if use_test else self.train_df
        pdf = df.toPandas()

        if instance_idx >= len(pdf):
            raise ValueError(f"instance_idx {instance_idx} out of range (max: {len(pdf)-1})")

        # Extract features
        from pyspark.ml.linalg import DenseVector, SparseVector

        row = pdf.iloc[instance_idx]
        vec = row[self.feature_col]
        if isinstance(vec, DenseVector):
            X = vec.toArray().reshape(1, -1)
        elif isinstance(vec, SparseVector):
            X = vec.toArray().reshape(1, -1)
        else:
            X = np.array(vec).reshape(1, -1)

        # Get SHAP values
        explainer = shap.TreeExplainer(self.booster)
        shap_values = explainer.shap_values(X)

        if isinstance(shap_values, list):
            shap_vals = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
        else:
            shap_vals = shap_values[0]

        # Get prediction
        pred_row = self.xgb_model.transform(df.limit(instance_idx + 1)).tail(1)[0]
        prediction = pred_row['prediction']
        probability = pred_row['probability'].toArray() if hasattr(pred_row['probability'], 'toArray') else pred_row['probability']

        # Build explanation
        contributions = []
        for i, (val, shap_val) in enumerate(zip(X[0], shap_vals)):
            # Get proper feature name from mapping
            feature_name = self.feature_idx_name_mapping.get(f'f{i}')
            if feature_name is None:
                feature_name = self.feature_names[i] if i < len(self.feature_names) else f'feature_{i}'
            contributions.append({
                'Feature': feature_name,
                'Value': val,
                'SHAP_Value': shap_val,
                'Direction': 'Positive' if shap_val > 0 else 'Negative',
                'Abs_Impact': abs(shap_val)
            })

        contributions_df = pd.DataFrame(contributions).sort_values('Abs_Impact', ascending=False)

        # Get actual label
        actual_label = row[self.problem.target]

        return {
            'instance_idx': instance_idx,
            'prediction': prediction,
            'probability': probability,
            'actual_label': actual_label,
            'base_value': explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value,
            'contributions': contributions_df,
            'top_positive': contributions_df[contributions_df['SHAP_Value'] > 0].head(5),
            'top_negative': contributions_df[contributions_df['SHAP_Value'] < 0].head(5)
        }

    def plot_shap_waterfall(self, instance_idx: int = 0, use_test: bool = True):
        """
        Plot a waterfall chart showing how each feature contributes to a prediction.

        Args:
            instance_idx: Index of the instance to explain
            use_test: Whether to use test set (True) or train set (False)
        """
        try:
            import shap
        except ImportError:
            raise ImportError(
                "SHAP is not installed. Install it with: pip install shap"
            )

        if self.xgb_model is None:
            raise ValueError("Model not trained. Call train_model() first.")

        # Get data
        df = self.test_df if use_test else self.train_df
        pdf = df.toPandas()

        # Extract features
        from pyspark.ml.linalg import DenseVector, SparseVector

        row = pdf.iloc[instance_idx]
        vec = row[self.feature_col]
        if isinstance(vec, DenseVector):
            X = vec.toArray().reshape(1, -1)
        elif isinstance(vec, SparseVector):
            X = vec.toArray().reshape(1, -1)
        else:
            X = np.array(vec).reshape(1, -1)

        # Get SHAP explanation
        explainer = shap.TreeExplainer(self.booster)
        shap_values = explainer(X)

        # Set feature names using the mapping
        feature_names = []
        for i in range(X.shape[1]):
            feature_name = self.feature_idx_name_mapping.get(f'f{i}')
            if feature_name is None:
                feature_name = self.feature_names[i] if i < len(self.feature_names) else f'feature_{i}'
            feature_names.append(feature_name)

        # Update shap_values feature names for proper labeling
        shap_values.feature_names = feature_names

        # Plot waterfall
        plt.figure(figsize=(12, 8))
        shap.plots.waterfall(shap_values[0], max_display=15, show=False)
        plt.title(f'SHAP Waterfall Plot - Instance {instance_idx}')
        plt.tight_layout()
        plt.savefig(f'shap_waterfall_instance_{instance_idx}.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"SHAP waterfall plot saved as 'shap_waterfall_instance_{instance_idx}.png'")

    def get_insight_analysis(
        self,
        schema_checks,
        min_support: float = 0.01,
        min_lift: float = 1.1,
        discover_microsegments: bool = True,
        plot: bool = True
    ):
        """
        Perform insight analysis with lift, support, and RIG metrics.

        This provides SparkBeyond-style feature insights showing which
        feature conditions have the highest lift for the target class.

        Args:
            schema_checks: SchemaChecks instance with column type info
            min_support: Minimum support threshold (fraction)
            min_lift: Minimum lift threshold
            discover_microsegments: Whether to discover feature combinations
            plot: Whether to generate plots

        Returns:
            Dictionary with insights DataFrame and full analysis result
        """
        from backend.core.features.insight_analyzer import FeatureInsightAnalyzer

        # Use original data (before transformation) for insight analysis
        analyzer = FeatureInsightAnalyzer(
            df=self.transformed_df,
            problem=self.problem,
            schema_checks=schema_checks,
            min_support=min_support,
            min_lift=min_lift
        )

        result = analyzer.get_analysis_result(discover_microsegments=discover_microsegments)

        if plot:
            analyzer.plot_lift_support_scatter(save_path='insight_lift_support.png')
            analyzer.plot_top_insights(save_path='insight_top_features.png')

        return {
            'insights_df': analyzer.to_dataframe(),
            'result': result,
            'analyzer': analyzer
        }
