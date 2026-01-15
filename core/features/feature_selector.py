"""XGboost based implementation""" 
from core.discovery import Problem, ProblemType
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
        self.feature_names = feature_names
        self.feature_col = feature_col
        self.random_state = random_state
        
        self.xgb_model = None
        self.train_df = None
        self.test_df = None

        self.feature_idx_name_mapping = feature_idx_name_mapping
        self.booster = None

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
        feature_importances = self.xgb_model.get_feature_importances()
        feature_importance_tags = list(feature_importances)

        importance_dict = {}
        for i, feature_name in enumerate(self.feature_names):
            if i < len(feature_importances):
                importance_dict[feature_name] = feature_importances[feature_importance_tags[i]]
            else:
                importance_dict[feature_name] = 0.0

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
        Get a summary of how each split affects classification probability
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



    
