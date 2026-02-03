import { useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { Link } from 'react-router-dom';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Label } from '@/components/ui/label';
import { Slider } from '@/components/ui/slider';
import { Progress } from '@/components/ui/progress';
import { dataApi, modelsApi } from '@/lib/api';
import type { TrainingMetrics } from '@/lib/api';
import {
  Brain,
  CheckCircle2,
  AlertCircle,
  Loader2,
  ArrowRight,
  TrendingUp,
  Target,
  Gauge,
  Activity,
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

export function ModelTraining() {
  const queryClient = useQueryClient();

  const [trainSplit, setTrainSplit] = useState([80]);
  const [maxDepth, setMaxDepth] = useState([4]);
  const [learningRate, setLearningRate] = useState([0.1]);
  const [numRounds, setNumRounds] = useState([100]);

  const { data: stateResponse } = useQuery({
    queryKey: ['pipelineState'],
    queryFn: async () => {
      const response = await dataApi.getState();
      return response.data;
    },
  });

  const state = stateResponse?.data;

  const trainMutation = useMutation({
    mutationFn: () =>
      modelsApi.train({
        train_split: trainSplit[0] / 100,
        max_depth: maxDepth[0],
        learning_rate: learningRate[0],
        num_rounds: numRounds[0],
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['pipelineState'] });
    },
  });

  const trainResult = trainMutation.data?.data?.data as {
    train_metrics: TrainingMetrics;
    test_metrics: TrainingMetrics;
    mlflow_run_id?: string;
  } | undefined;

  if (!state?.has_preprocessed) {
    return (
      <div className="space-y-6">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Model Training</h1>
          <p className="text-muted-foreground mt-2">
            Train your XGBoost model with customized hyperparameters
          </p>
        </div>
        <Card className="border-yellow-500/50 bg-yellow-50/50 dark:bg-yellow-950/20">
          <CardContent className="pt-6">
            <div className="flex items-center gap-4">
              <AlertCircle className="w-8 h-8 text-yellow-600" />
              <div>
                <h3 className="font-semibold">Complete Previous Steps First</h3>
                <p className="text-muted-foreground">
                  Please complete feature engineering and preprocessing first.
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

  const metricsChartData = trainResult
    ? [
        {
          name: 'Accuracy',
          Train: (trainResult.train_metrics.accuracy || 0) * 100,
          Test: (trainResult.test_metrics.accuracy || 0) * 100,
        },
        {
          name: 'Precision',
          Train: (trainResult.train_metrics.precision || 0) * 100,
          Test: (trainResult.test_metrics.precision || 0) * 100,
        },
        {
          name: 'Recall',
          Train: (trainResult.train_metrics.recall || 0) * 100,
          Test: (trainResult.test_metrics.recall || 0) * 100,
        },
        {
          name: 'F1 Score',
          Train: (trainResult.train_metrics.f1_score || 0) * 100,
          Test: (trainResult.test_metrics.f1_score || 0) * 100,
        },
        {
          name: 'AUC-ROC',
          Train: (trainResult.train_metrics.auc_roc || 0) * 100,
          Test: (trainResult.test_metrics.auc_roc || 0) * 100,
        },
      ]
    : [];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Model Training</h1>
        <p className="text-muted-foreground mt-2">
          Train your XGBoost model with customized hyperparameters
        </p>
      </div>

      {/* Training Configuration */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Brain className="w-5 h-5" />
            Training Configuration
          </CardTitle>
          <CardDescription>
            Configure XGBoost hyperparameters for training
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="space-y-3">
              <div className="flex justify-between">
                <Label className="font-medium">Training Split</Label>
                <span className="text-sm text-muted-foreground">{trainSplit[0]}%</span>
              </div>
              <Slider
                value={trainSplit}
                onValueChange={setTrainSplit}
                min={50}
                max={95}
                step={5}
              />
              <p className="text-xs text-muted-foreground">
                Percentage of data used for training
              </p>
            </div>

            <div className="space-y-3">
              <div className="flex justify-between">
                <Label className="font-medium">Max Depth</Label>
                <span className="text-sm text-muted-foreground">{maxDepth[0]}</span>
              </div>
              <Slider
                value={maxDepth}
                onValueChange={setMaxDepth}
                min={2}
                max={10}
                step={1}
              />
              <p className="text-xs text-muted-foreground">
                Maximum tree depth (higher = more complex)
              </p>
            </div>

            <div className="space-y-3">
              <div className="flex justify-between">
                <Label className="font-medium">Learning Rate</Label>
                <span className="text-sm text-muted-foreground">{learningRate[0]}</span>
              </div>
              <Slider
                value={learningRate}
                onValueChange={setLearningRate}
                min={0.01}
                max={0.5}
                step={0.01}
              />
              <p className="text-xs text-muted-foreground">
                Step size shrinkage (lower = more robust)
              </p>
            </div>

            <div className="space-y-3">
              <div className="flex justify-between">
                <Label className="font-medium">Number of Rounds</Label>
                <span className="text-sm text-muted-foreground">{numRounds[0]}</span>
              </div>
              <Slider
                value={numRounds}
                onValueChange={setNumRounds}
                min={10}
                max={500}
                step={10}
              />
              <p className="text-xs text-muted-foreground">
                Number of boosting iterations
              </p>
            </div>
          </div>

          <Button
            onClick={() => trainMutation.mutate()}
            disabled={trainMutation.isPending}
            className="w-full md:w-auto"
            size="lg"
          >
            {trainMutation.isPending ? (
              <>
                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                Training...
              </>
            ) : (
              <>
                <Brain className="w-4 h-4 mr-2" />
                Train Model
              </>
            )}
          </Button>

          {trainMutation.isPending && (
            <div className="space-y-2">
              <p className="text-sm text-muted-foreground">Training in progress...</p>
              <Progress value={undefined} className="h-2" />
            </div>
          )}
        </CardContent>
      </Card>

      {/* Training Results */}
      {trainResult && (
        <>
          <div className="flex items-center gap-2 text-green-600">
            <CheckCircle2 className="w-5 h-5" />
            <span className="font-medium">Model trained successfully!</span>
            {trainResult?.mlflow_run_id && (
              <span className="text-xs text-muted-foreground ml-2">
                MLflow run: <code>{trainResult.mlflow_run_id.slice(0, 8)}...</code>
              </span>
            )}
          </div>

          {/* Metrics Cards */}
          <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
            <Card>
              <CardContent className="pt-6">
                <div className="text-center">
                  <Target className="w-8 h-8 mx-auto text-blue-500 mb-2" />
                  <div className="text-2xl font-bold">
                    {((trainResult.test_metrics.accuracy || 0) * 100).toFixed(1)}%
                  </div>
                  <div className="text-sm text-muted-foreground">Accuracy</div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="pt-6">
                <div className="text-center">
                  <Gauge className="w-8 h-8 mx-auto text-green-500 mb-2" />
                  <div className="text-2xl font-bold">
                    {((trainResult.test_metrics.precision || 0) * 100).toFixed(1)}%
                  </div>
                  <div className="text-sm text-muted-foreground">Precision</div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="pt-6">
                <div className="text-center">
                  <Activity className="w-8 h-8 mx-auto text-purple-500 mb-2" />
                  <div className="text-2xl font-bold">
                    {((trainResult.test_metrics.recall || 0) * 100).toFixed(1)}%
                  </div>
                  <div className="text-sm text-muted-foreground">Recall</div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="pt-6">
                <div className="text-center">
                  <TrendingUp className="w-8 h-8 mx-auto text-orange-500 mb-2" />
                  <div className="text-2xl font-bold">
                    {((trainResult.test_metrics.f1_score || 0) * 100).toFixed(1)}%
                  </div>
                  <div className="text-sm text-muted-foreground">F1 Score</div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="pt-6">
                <div className="text-center">
                  <Brain className="w-8 h-8 mx-auto text-red-500 mb-2" />
                  <div className="text-2xl font-bold">
                    {((trainResult.test_metrics.auc_roc || 0) * 100).toFixed(1)}%
                  </div>
                  <div className="text-sm text-muted-foreground">AUC-ROC</div>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Metrics Chart */}
          <Card>
            <CardHeader>
              <CardTitle>Train vs Test Performance</CardTitle>
              <CardDescription>
                Compare metrics between training and test sets
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={metricsChartData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="name" />
                    <YAxis domain={[0, 100]} />
                    <Tooltip
                      formatter={(value) => typeof value === 'number' ? `${value.toFixed(1)}%` : value}
                    />
                    <Legend />
                    <Bar dataKey="Train" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                    <Bar dataKey="Test" fill="#10b981" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>

          {/* Next Steps */}
          <Card className="border-green-500/50 bg-green-50/50 dark:bg-green-950/20">
            <CardContent className="pt-6">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-4">
                  <CheckCircle2 className="w-8 h-8 text-green-600" />
                  <div>
                    <h3 className="font-semibold">Model Ready!</h3>
                    <p className="text-muted-foreground">
                      View feature importance and insights from your trained model.
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
        </>
      )}
    </div>
  );
}
