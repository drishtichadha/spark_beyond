from backend.core.metrics.base import BaseMetrics

class TargetAggregation(BaseMetrics):
    def __init__(self, spark):
        super().__init__(spark)

    