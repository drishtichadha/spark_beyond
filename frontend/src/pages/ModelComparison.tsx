import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { Link } from 'react-router-dom';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { dataApi, modelsApi } from '@/lib/api';
import type { FeatureComparisonResult } from '@/lib/api';
import {
  BarChart3,
  AlertCircle,
  ArrowRight,
  Loader2,
  Trophy,
  Zap,
  Brain,
  GitCompare,
  TrendingUp,
  Layers,
  ArrowUpRight,
  ArrowDownRight,
} from 'lucide-react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';

interface BaselineResult {
  model_name: string;
  metrics: Record<string, number>;
  training_time: number;
}

interface AutoMLResult {
  best_score: number;
  problem_type: string;
  metric: string;
  search_time: number;
  feature_importance?: Array<{ feature: string; importance: number }>;
}

export function ModelComparison() {
  const queryClient = useQueryClient();

  const { data: stateResponse } = useQuery({
    queryKey: ['pipelineState'],
    queryFn: async () => {
      const response = await dataApi.getState();
      return response.data;
    },
  });

  const state = stateResponse?.data;

  const baselinesMutation = useMutation({
    mutationFn: () => modelsApi.trainBaselines(),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['baselines'] });
    },
  });

  const automlMutation = useMutation({
    mutationFn: () =>
      modelsApi.runAutoML({
        timeout: 120,
        cpu_limit: 4,
        quick_mode: true,
      }),
  });

  const featureComparisonMutation = useMutation({
    mutationFn: () => modelsApi.compareFeatures(),
  });

  const baselines = baselinesMutation.data?.data?.data as BaselineResult[] | undefined;
  const automlResult = automlMutation.data?.data?.data as AutoMLResult | undefined;
  const featureComparison = featureComparisonMutation.data?.data?.data as FeatureComparisonResult | undefined;

  if (!state?.has_preprocessed) {
    return (
      <div className="space-y-6">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Model Comparison</h1>
          <p className="text-muted-foreground mt-2">
            Compare baseline models and run AutoML
          </p>
        </div>
        <Card className="border-yellow-500/50 bg-yellow-50/50 dark:bg-yellow-950/20">
          <CardContent className="pt-6">
            <div className="flex items-center gap-4">
              <AlertCircle className="w-8 h-8 text-yellow-600" />
              <div>
                <h3 className="font-semibold">Complete Feature Engineering First</h3>
                <p className="text-muted-foreground">
                  Please preprocess your features before comparing models.
                </p>
                <Link to="/features">
                  <Button className="mt-2" variant="outline">
                    Go to Feature Engineering <ArrowRight className="w-4 h-4 ml-2" />
                  </Button>
                </Link>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  // Prepare chart data
  const chartData = baselines?.map((result) => ({
    name: result.model_name.length > 20
      ? result.model_name.slice(0, 20) + '...'
      : result.model_name,
    Accuracy: (result.metrics.accuracy || 0) * 100,
    F1: (result.metrics.f1 || 0) * 100,
    Precision: (result.metrics.precision || 0) * 100,
    Recall: (result.metrics.recall || 0) * 100,
  })) || [];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Model Comparison</h1>
        <p className="text-muted-foreground mt-2">
          Compare baseline models and explore AutoML options
        </p>
      </div>

      {/* Baseline Models */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <GitCompare className="w-5 h-5" />
            Baseline Models
          </CardTitle>
          <CardDescription>
            Train simple baseline models to establish performance benchmarks
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="flex items-center gap-4">
            <Button
              onClick={() => baselinesMutation.mutate()}
              disabled={baselinesMutation.isPending}
            >
              {baselinesMutation.isPending ? (
                <>
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  Training Baselines...
                </>
              ) : (
                <>
                  <Brain className="w-4 h-4 mr-2" />
                  Train Baseline Models
                </>
              )}
            </Button>
            <p className="text-sm text-muted-foreground">
              Trains Naive, Decision Tree, and Logistic Regression
            </p>
          </div>

          {baselinesMutation.isPending && (
            <div className="space-y-2">
              <p className="text-sm text-muted-foreground">Training in progress...</p>
              <Progress value={undefined} className="h-2" />
            </div>
          )}

          {baselines && baselines.length > 0 && (
            <>
              {/* Results Table */}
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b">
                      <th className="text-left py-3 px-4">Model</th>
                      <th className="text-right py-3 px-4">Accuracy</th>
                      <th className="text-right py-3 px-4">Precision</th>
                      <th className="text-right py-3 px-4">Recall</th>
                      <th className="text-right py-3 px-4">F1</th>
                      <th className="text-right py-3 px-4">Time</th>
                    </tr>
                  </thead>
                  <tbody>
                    {baselines.map((result, i) => (
                      <tr key={i} className="border-b hover:bg-slate-50 dark:hover:bg-slate-900">
                        <td className="py-3 px-4 font-medium">{result.model_name}</td>
                        <td className="text-right py-3 px-4">
                          {((result.metrics.accuracy || 0) * 100).toFixed(1)}%
                        </td>
                        <td className="text-right py-3 px-4">
                          {((result.metrics.precision || 0) * 100).toFixed(1)}%
                        </td>
                        <td className="text-right py-3 px-4">
                          {((result.metrics.recall || 0) * 100).toFixed(1)}%
                        </td>
                        <td className="text-right py-3 px-4">
                          {((result.metrics.f1 || 0) * 100).toFixed(1)}%
                        </td>
                        <td className="text-right py-3 px-4">
                          {result.training_time.toFixed(1)}s
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {/* Chart */}
              <div className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="name" tick={{ fontSize: 12 }} />
                    <YAxis domain={[0, 100]} />
                    <Tooltip formatter={(value) => typeof value === 'number' ? `${value.toFixed(1)}%` : value} />
                    <Legend />
                    <Bar dataKey="Accuracy" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                    <Bar dataKey="F1" fill="#10b981" radius={[4, 4, 0, 0]} />
                    <Bar dataKey="Precision" fill="#f59e0b" radius={[4, 4, 0, 0]} />
                    <Bar dataKey="Recall" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>

              {/* Best Model */}
              {baselines.length > 0 && (
                <Card className="bg-green-50 dark:bg-green-950/30 border-green-500/50">
                  <CardContent className="pt-4">
                    <div className="flex items-center gap-3">
                      <Trophy className="w-8 h-8 text-green-600" />
                      <div>
                        <p className="text-sm text-muted-foreground">Best Baseline</p>
                        <p className="font-semibold">
                          {baselines.reduce((best, current) =>
                            (current.metrics.accuracy || 0) > (best.metrics.accuracy || 0)
                              ? current
                              : best
                          ).model_name}
                        </p>
                      </div>
                      <Badge variant="success" className="ml-auto">
                        {(
                          baselines.reduce((best, current) =>
                            (current.metrics.accuracy || 0) > (best.metrics.accuracy || 0)
                              ? current
                              : best
                          ).metrics.accuracy! * 100
                        ).toFixed(1)}%
                        Accuracy
                      </Badge>
                    </div>
                  </CardContent>
                </Card>
              )}
            </>
          )}
        </CardContent>
      </Card>

      {/* Feature Engineering Impact */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Layers className="w-5 h-5" />
            Feature Engineering Impact
          </CardTitle>
          <CardDescription>
            Compare model performance with base features vs engineered features
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="p-4 bg-purple-50 dark:bg-purple-950/30 rounded-lg border border-purple-200 dark:border-purple-800">
            <p className="text-sm">
              <strong>What this does:</strong> Trains models (Decision Tree & XGBoost) on both
              original features and engineered features to measure the impact of feature engineering.
            </p>
          </div>

          <Button
            onClick={() => featureComparisonMutation.mutate()}
            disabled={featureComparisonMutation.isPending || !state?.has_features}
            variant="secondary"
          >
            {featureComparisonMutation.isPending ? (
              <>
                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                Comparing Features... (This may take a few minutes)
              </>
            ) : (
              <>
                <TrendingUp className="w-4 h-4 mr-2" />
                Compare Base vs Engineered Features
              </>
            )}
          </Button>

          {!state?.has_features && (
            <p className="text-sm text-muted-foreground">
              Generate features first to enable this comparison.
            </p>
          )}

          {featureComparisonMutation.isPending && (
            <div className="space-y-2">
              <p className="text-sm text-muted-foreground">
                Training models on both feature sets...
              </p>
              <Progress value={undefined} className="h-2" />
            </div>
          )}

          {featureComparisonMutation.isError && (
            <div className="flex items-center gap-2 p-3 bg-red-50 dark:bg-red-950/30 rounded-lg border border-red-200 dark:border-red-800">
              <AlertCircle className="w-5 h-5 text-red-500" />
              <span className="text-sm text-red-700 dark:text-red-300">
                Failed to compare features
              </span>
            </div>
          )}

          {featureComparison && (
            <div className="space-y-6">
              {/* Improvement Summary */}
              {featureComparison.improvements?.feature_engineering && (
                <Card className="bg-gradient-to-r from-green-50 to-emerald-50 dark:from-green-950/30 dark:to-emerald-950/30 border-green-500/50">
                  <CardContent className="pt-4">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        {featureComparison.improvements.feature_engineering.percentage_improvement > 0 ? (
                          <ArrowUpRight className="w-8 h-8 text-green-600" />
                        ) : (
                          <ArrowDownRight className="w-8 h-8 text-red-600" />
                        )}
                        <div>
                          <p className="text-sm text-muted-foreground">Feature Engineering Impact</p>
                          <p className="text-2xl font-bold">
                            {featureComparison.improvements.feature_engineering.percentage_improvement > 0 ? '+' : ''}
                            {featureComparison.improvements.feature_engineering.percentage_improvement.toFixed(2)}%
                          </p>
                        </div>
                      </div>
                      <div className="text-right">
                        <p className="text-sm text-muted-foreground">
                          {featureComparison.improvements.feature_engineering.original_score.toFixed(3)} →{' '}
                          {featureComparison.improvements.feature_engineering.engineered_score.toFixed(3)}
                        </p>
                        <p className="text-xs text-muted-foreground mt-1">
                          Using {featureComparison.improvements.feature_engineering.engineered_model}
                        </p>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              )}

              {/* Comparison Table */}
              <Tabs defaultValue="table">
                <TabsList>
                  <TabsTrigger value="table">Comparison Table</TabsTrigger>
                  <TabsTrigger value="chart">Visual Comparison</TabsTrigger>
                </TabsList>

                <TabsContent value="table" className="mt-4">
                  <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                      <thead>
                        <tr className="border-b">
                          <th className="text-left py-3 px-4">Model</th>
                          <th className="text-left py-3 px-4">Feature Set</th>
                          <th className="text-right py-3 px-4">Accuracy</th>
                          <th className="text-right py-3 px-4">Precision</th>
                          <th className="text-right py-3 px-4">Recall</th>
                          <th className="text-right py-3 px-4">F1</th>
                          <th className="text-right py-3 px-4">Features</th>
                          <th className="text-right py-3 px-4">Time</th>
                        </tr>
                      </thead>
                      <tbody>
                        {featureComparison.comparison_table?.map((exp, i) => (
                          <tr
                            key={i}
                            className={`border-b hover:bg-slate-50 dark:hover:bg-slate-900 ${
                              exp.feature_set === 'engineered' ? 'bg-green-50/50 dark:bg-green-950/20' : ''
                            }`}
                          >
                            <td className="py-3 px-4 font-medium">{exp.model}</td>
                            <td className="py-3 px-4">
                              <Badge variant={exp.feature_set === 'engineered' ? 'default' : 'outline'}>
                                {exp.feature_set}
                              </Badge>
                            </td>
                            <td className="text-right py-3 px-4">
                              {((exp.accuracy || 0) * 100).toFixed(1)}%
                            </td>
                            <td className="text-right py-3 px-4">
                              {((exp.precision || 0) * 100).toFixed(1)}%
                            </td>
                            <td className="text-right py-3 px-4">
                              {((exp.recall || 0) * 100).toFixed(1)}%
                            </td>
                            <td className="text-right py-3 px-4">
                              {((exp.f1 || 0) * 100).toFixed(1)}%
                            </td>
                            <td className="text-right py-3 px-4">{exp.num_features || '-'}</td>
                            <td className="text-right py-3 px-4">{exp.training_time?.toFixed(1)}s</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </TabsContent>

                <TabsContent value="chart" className="mt-4">
                  <div className="h-80">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart
                        data={featureComparison.comparison_table?.map((exp) => ({
                          name: `${exp.model}\n(${exp.feature_set})`,
                          Accuracy: (exp.accuracy || 0) * 100,
                          F1: (exp.f1 || 0) * 100,
                          Precision: (exp.precision || 0) * 100,
                          Recall: (exp.recall || 0) * 100,
                        }))}
                      >
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="name" tick={{ fontSize: 10 }} interval={0} />
                        <YAxis domain={[0, 100]} />
                        <Tooltip formatter={(value) => typeof value === 'number' ? `${value.toFixed(1)}%` : value} />
                        <Legend />
                        <Bar dataKey="Accuracy" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                        <Bar dataKey="F1" fill="#10b981" radius={[4, 4, 0, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </TabsContent>
              </Tabs>

              {/* Best Model Card */}
              {featureComparison.best_model && (
                <Card className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-950/30 dark:to-indigo-950/30 border-blue-500/50">
                  <CardContent className="pt-4">
                    <div className="flex items-center gap-3">
                      <Trophy className="w-8 h-8 text-blue-600" />
                      <div>
                        <p className="text-sm text-muted-foreground">Best Performing Model</p>
                        <p className="font-semibold">{featureComparison.best_model.name}</p>
                      </div>
                      <Badge variant="default" className="ml-auto">
                        {featureComparison.best_model.feature_set}
                      </Badge>
                      <Badge variant="success">
                        {((featureComparison.best_model.metrics?.accuracy || 0) * 100).toFixed(1)}% Accuracy
                      </Badge>
                    </div>
                  </CardContent>
                </Card>
              )}
            </div>
          )}
        </CardContent>
      </Card>

      {/* AutoML */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Zap className="w-5 h-5" />
            AutoML with LightAutoML
          </CardTitle>
          <CardDescription>
            Automatically search for the best model and hyperparameters
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="p-4 bg-blue-50 dark:bg-blue-950/30 rounded-lg border border-blue-200 dark:border-blue-800">
            <p className="text-sm">
              <strong>Note:</strong> AutoML will automatically try multiple algorithms
              (LightGBM, CatBoost, Linear models) and find the optimal configuration.
              This may take a few minutes.
            </p>
          </div>

          <Button
            onClick={() => automlMutation.mutate()}
            disabled={automlMutation.isPending}
            variant="secondary"
          >
            {automlMutation.isPending ? (
              <>
                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                Running AutoML... (This may take a few minutes)
              </>
            ) : (
              <>
                <Zap className="w-4 h-4 mr-2" />
                Run AutoML Search
              </>
            )}
          </Button>

          {automlMutation.isPending && (
            <div className="space-y-2">
              <p className="text-sm text-muted-foreground">
                Searching for the best model configuration...
              </p>
              <Progress value={undefined} className="h-2" />
            </div>
          )}

          {automlResult && (
            <div className="space-y-4">
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <Card>
                  <CardContent className="pt-4">
                    <div className="text-center">
                      <div className="text-2xl font-bold text-blue-600">
                        {(automlResult.best_score * 100).toFixed(1)}%
                      </div>
                      <div className="text-sm text-muted-foreground">Best Score</div>
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardContent className="pt-4">
                    <div className="text-center">
                      <div className="text-2xl font-bold text-purple-600">
                        {automlResult.problem_type}
                      </div>
                      <div className="text-sm text-muted-foreground">Problem Type</div>
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardContent className="pt-4">
                    <div className="text-center">
                      <div className="text-2xl font-bold text-green-600">
                        {automlResult.metric}
                      </div>
                      <div className="text-sm text-muted-foreground">Metric</div>
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardContent className="pt-4">
                    <div className="text-center">
                      <div className="text-2xl font-bold text-orange-600">
                        {automlResult.search_time.toFixed(1)}s
                      </div>
                      <div className="text-sm text-muted-foreground">Search Time</div>
                    </div>
                  </CardContent>
                </Card>
              </div>

              {automlResult.feature_importance && (
                <div>
                  <h4 className="font-semibold mb-3">AutoML Feature Importance</h4>
                  <div className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart
                        data={automlResult.feature_importance.slice(0, 10)}
                        layout="vertical"
                        margin={{ left: 120 }}
                      >
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis type="number" />
                        <YAxis dataKey="feature" type="category" tick={{ fontSize: 11 }} />
                        <Tooltip />
                        <Bar dataKey="importance" fill="#8b5cf6" radius={[0, 4, 4, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              )}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Next Steps */}
      {(baselines || automlResult || featureComparison) && (
        <Card className="border-blue-500/50 bg-blue-50/50 dark:bg-blue-950/20">
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-4">
                <BarChart3 className="w-8 h-8 text-blue-600" />
                <div>
                  <h3 className="font-semibold">Ready to Analyze!</h3>
                  <p className="text-muted-foreground">
                    View detailed insights and feature analysis.
                  </p>
                </div>
              </div>
              <Link to="/insights">
                <Button>
                  View Insights <ArrowRight className="w-4 h-4 ml-2" />
                </Button>
              </Link>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
