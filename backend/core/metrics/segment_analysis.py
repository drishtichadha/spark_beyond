import itertools
import json
from datetime import date
from typing import List, Tuple

import polars as pl

from kpi_analytics.base.general import Filter
from kpi_analytics.base.metrics import BaseMetrics


class SegmentAnalysis:
    def __init__(
        self,
        metric_class: BaseMetrics,
        user_id: str,
        dimensions: List[str],
        baseline_filters: List[Filter] = None,
        baseline_date_range: Tuple[date, date] = None,
        comparison_filters: List[Filter] = None,
        comparison_date_range: Tuple[date, date] = None,
    ) -> None:
        self.baseline_metric = metric_class(user_id=user_id)
        self.comparison_metric = metric_class(user_id=user_id)
        self.metric_name = self.baseline_metric.metric_name
        self.dimensions = dimensions
        self.baseline_filters = baseline_filters
        self.baseline_date_range = baseline_date_range
        self.is_comparison = (
            True if any([comparison_filters, comparison_date_range]) else False
        )
        self.comparison_filters = comparison_filters
        self.comparison_date_range = comparison_date_range

    def get_dimension_combinations(self, rca_depth: int = None):
        all_combinations = [
            comb
            for r in range(1, rca_depth + 1)
            for comb in itertools.combinations(self.dimensions, r)
        ]
        print(all_combinations)
        return all_combinations

    def get_overall_metrics(self):
        baseline_overall = json.loads(
            self.baseline_metric.calculate(
                filters=self.baseline_filters, date_range=self.baseline_date_range
            )
        )
        comparison_overall = None

        if self.is_comparison:
            comparison_overall = json.loads(
                self.comparison_metric.calculate(
                    filters=self.comparison_filters,
                    date_range=self.comparison_date_range,
                )
            )
        return {
            "baseline": baseline_overall,
            "comparison": comparison_overall,
        }

    def get_segment_analysis(self):
        all_dim_combinations = self.get_dimension_combinations(rca_depth=1)

        results = {"baseline": []}
        if self.is_comparison:
            results["comparison"] = []
        for combi in all_dim_combinations:
            results["baseline"].append(
                {
                    "<dim>".join(list(combi)): json.loads(
                        self.baseline_metric.calculate(list(combi))
                    )
                }
            )
            if self.is_comparison:
                results["comparison"].append(
                    {
                        "<dim>".join(list(combi)): json.loads(
                            self.comparison_metric.calculate(list(combi))
                        )
                    }
                )

        return results

    def create_merge_data(self, combined_segment_data):

        baseline_df = pl.DataFrame(combined_segment_data["baseline"])
        if self.is_comparison:
            comparison_df = pl.DataFrame(combined_segment_data["comparison"])
            baseline_df = baseline_df.join(
                comparison_df, how="outer", on="segment", suffix="_comparison"
            )
        return baseline_df

    def combine_segments(self, segment_results):

        combined_segments = {"baseline": [], "comparison": []}
        for segment_type, dimension_list in segment_results.items():
            for dimension_segments in dimension_list:
                for dimension_tag, segments in dimension_segments.items():
                    dimensions = dimension_tag.split("<dim>")
                    # print(dimension_tag)
                    # print(segments)
                    for segment in segments:
                        # print(segment)
                        combined_segments[segment_type].append(
                            {
                                "segment": "<seg>".join(
                                    [
                                        f"{key}:{value}"
                                        for key, value in segment.items()
                                        if key in dimensions
                                    ]
                                ),
                                self.metric_name
                                + f"_{segment_type}": segment[self.metric_name],
                                f"size_{segment_type}": segment.get("size", 0),
                            }
                        )
        return combined_segments

    def get_impact_calc(self):
        agg = (
            (
                (
                    (
                        pl.col(self.metric_name + "_comparison")
                        - pl.col(self.metric_name + "_baseline")
                    )
                    / (
                        pl.col(self.metric_name + "_overall_comparison")
                        - pl.col(self.metric_name + "_overall_baseline")
                    )
                )
                * pl.col("size_pct_baseline")
            ).alias("impact"),
            (
                (
                    (
                        pl.col(self.metric_name + "_comparison")
                        - pl.col(self.metric_name + "_baseline")
                    )
                    / (
                        pl.col(self.metric_name + "_overall_comparison")
                        - pl.col(self.metric_name + "_overall_baseline")
                    )
                )
                * pl.col("size_pct_baseline")
            )
            .abs()
            .alias("abs_impact"),
        )
        return agg

    def analyze(self):
        overall_metric = self.get_overall_metrics()
        base_metric = overall_metric["baseline"][0][self.metric_name]
        base_size = overall_metric["baseline"][0]["size"]
        comp_metric = overall_metric["comparison"][0][self.metric_name]
        comp_size = overall_metric["comparison"][0]["size"]

        segments = self.get_segment_analysis()

        combined_segment_list = self.combine_segments(segments)
        combined_segment_df = self.create_merge_data(combined_segment_list)
        # print(overall_metric)
        combined_segment_df = combined_segment_df.with_columns(
            pl.lit(base_metric).alias(self.metric_name + "_overall_baseline"),
            pl.lit(base_size).alias("size_overall_baseline"),
        )
        combined_segment_df = combined_segment_df.with_columns(
            (
                pl.col(self.metric_name + "_baseline")
                / pl.col(self.metric_name + "_overall_baseline")
            ).alias("value_pct_baseline"),
            (pl.col("size_baseline") / pl.col("size_overall_baseline")).alias(
                "size_pct_baseline"
            ),
        )

        if self.is_comparison:
            combined_segment_df = combined_segment_df.with_columns(
                pl.lit(comp_metric).alias(self.metric_name + "_overall_comparison"),
                pl.lit(comp_size).alias("size_overall_comparison"),
            )
            combined_segment_df = combined_segment_df.with_columns(
                (
                    pl.col(self.metric_name + "_comparison")
                    / pl.col(self.metric_name + "_overall_comparison")
                ).alias("value_pct_comparison"),
                (pl.col("size_comparison") / pl.col("size_overall_comparison")).alias(
                    "size_pct_comparison"
                ),
                (
                    (
                        pl.col(self.metric_name + "_comparison")
                        / pl.col(self.metric_name + "_baseline")
                    )
                    - 1
                ).alias(self.metric_name + "_change"),
                ((pl.col("size_comparison") / pl.col("size_baseline")) - 1).alias(
                    "size_change"
                ),
            )
            combined_segment_df = combined_segment_df.with_columns(
                self.get_impact_calc()
            )

        combined_segment_df = combined_segment_df.write_json(row_oriented=True)
        return combined_segment_df

