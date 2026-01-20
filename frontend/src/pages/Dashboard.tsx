import { useQuery } from '@tanstack/react-query';
import { Link } from 'react-router-dom';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { dataApi } from '@/lib/api';
import type { PipelineState } from '@/lib/api';
import {
  Database,
  Sparkles,
  Brain,
  Lightbulb,
  ArrowRight,
  CheckCircle2,
  Loader2,
} from 'lucide-react';

const steps = [
  {
    id: 1,
    name: 'Load Data',
    description: 'Upload or select your dataset',
    href: '/data',
    icon: Database,
    stateKey: 'has_data' as keyof PipelineState,
  },
  {
    id: 2,
    name: 'Define Problem',
    description: 'Set target and problem type',
    href: '/data',
    icon: Database,
    stateKey: 'has_problem' as keyof PipelineState,
  },
  {
    id: 3,
    name: 'Generate Features',
    description: 'Create engineered features',
    href: '/features',
    icon: Sparkles,
    stateKey: 'has_features' as keyof PipelineState,
  },
  {
    id: 4,
    name: 'Preprocess',
    description: 'Encode and transform features',
    href: '/features',
    icon: Sparkles,
    stateKey: 'has_preprocessed' as keyof PipelineState,
  },
  {
    id: 5,
    name: 'Train Model',
    description: 'Train XGBoost classifier',
    href: '/training',
    icon: Brain,
    stateKey: 'has_model' as keyof PipelineState,
  },
  {
    id: 6,
    name: 'Analyze Results',
    description: 'View insights and importance',
    href: '/insights',
    icon: Lightbulb,
    stateKey: 'has_model' as keyof PipelineState,
  },
];

export function Dashboard() {
  const { data: stateResponse, isLoading } = useQuery({
    queryKey: ['pipelineState'],
    queryFn: async () => {
      const response = await dataApi.getState();
      return response.data;
    },
    refetchInterval: 5000,
  });

  const state = stateResponse?.data;

  const completedSteps = state
    ? steps.filter((step) => state[step.stateKey]).length
    : 0;
  const progress = (completedSteps / steps.length) * 100;

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Dashboard</h1>
        <p className="text-muted-foreground mt-2">
          Welcome to Spark Tune - ML Feature Discovery Platform
        </p>
      </div>

      {/* Progress Overview */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            <span>Pipeline Progress</span>
            <Badge variant={completedSteps === steps.length ? 'success' : 'secondary'}>
              {completedSteps} / {steps.length} steps
            </Badge>
          </CardTitle>
          <CardDescription>
            Complete all steps to train your ML model
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Progress value={progress} className="h-2" />
          <p className="text-sm text-muted-foreground mt-2">
            {progress.toFixed(0)}% complete
          </p>
        </CardContent>
      </Card>

      {/* Pipeline Steps */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
        {steps.map((step, index) => {
          const isCompleted = state?.[step.stateKey] ?? false;
          const isPrevCompleted =
            index === 0 || (state?.[steps[index - 1].stateKey] ?? false);
          const isAvailable = isPrevCompleted;

          return (
            <Card
              key={step.id}
              className={`transition-all ${
                isCompleted
                  ? 'border-green-500/50 bg-green-50/50 dark:bg-green-950/20'
                  : isAvailable
                  ? 'hover:border-blue-500/50 hover:shadow-md'
                  : 'opacity-50'
              }`}
            >
              <CardHeader className="pb-3">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-3">
                    <div
                      className={`w-10 h-10 rounded-full flex items-center justify-center ${
                        isCompleted
                          ? 'bg-green-500 text-white'
                          : 'bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-400'
                      }`}
                    >
                      {isCompleted ? (
                        <CheckCircle2 className="w-5 h-5" />
                      ) : (
                        <step.icon className="w-5 h-5" />
                      )}
                    </div>
                    <div>
                      <CardTitle className="text-base">
                        {step.id}. {step.name}
                      </CardTitle>
                      <CardDescription className="text-xs">
                        {step.description}
                      </CardDescription>
                    </div>
                  </div>
                </div>
              </CardHeader>
              <CardContent className="pt-0">
                {isAvailable && !isCompleted && (
                  <Link to={step.href}>
                    <Button size="sm" className="w-full">
                      Start <ArrowRight className="w-4 h-4 ml-2" />
                    </Button>
                  </Link>
                )}
                {isCompleted && (
                  <Link to={step.href}>
                    <Button size="sm" variant="outline" className="w-full">
                      Review <ArrowRight className="w-4 h-4 ml-2" />
                    </Button>
                  </Link>
                )}
              </CardContent>
            </Card>
          );
        })}
      </div>

      {/* Dataset Info */}
      {state?.dataset_info && (
        <Card>
          <CardHeader>
            <CardTitle>Current Dataset</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="text-center p-4 bg-slate-50 dark:bg-slate-900 rounded-lg">
                <div className="text-2xl font-bold text-blue-600">
                  {state.dataset_info.rows.toLocaleString()}
                </div>
                <div className="text-sm text-muted-foreground">Rows</div>
              </div>
              <div className="text-center p-4 bg-slate-50 dark:bg-slate-900 rounded-lg">
                <div className="text-2xl font-bold text-purple-600">
                  {state.dataset_info.columns}
                </div>
                <div className="text-sm text-muted-foreground">Columns</div>
              </div>
              {state.problem && (
                <>
                  <div className="text-center p-4 bg-slate-50 dark:bg-slate-900 rounded-lg">
                    <div className="text-2xl font-bold text-green-600">
                      {state.problem.target}
                    </div>
                    <div className="text-sm text-muted-foreground">Target</div>
                  </div>
                  <div className="text-center p-4 bg-slate-50 dark:bg-slate-900 rounded-lg">
                    <Badge className="text-lg px-3 py-1">
                      {state.problem.type}
                    </Badge>
                    <div className="text-sm text-muted-foreground mt-1">
                      Problem Type
                    </div>
                  </div>
                </>
              )}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Quick Actions */}
      <Card>
        <CardHeader>
          <CardTitle>Quick Actions</CardTitle>
        </CardHeader>
        <CardContent className="flex flex-wrap gap-3">
          <Link to="/data">
            <Button>
              <Database className="w-4 h-4 mr-2" />
              Load Dataset
            </Button>
          </Link>
          <Link to="/features">
            <Button variant="outline">
              <Sparkles className="w-4 h-4 mr-2" />
              Engineer Features
            </Button>
          </Link>
          <Link to="/training">
            <Button variant="outline">
              <Brain className="w-4 h-4 mr-2" />
              Train Model
            </Button>
          </Link>
          <Link to="/insights">
            <Button variant="outline">
              <Lightbulb className="w-4 h-4 mr-2" />
              View Insights
            </Button>
          </Link>
        </CardContent>
      </Card>

      {isLoading && (
        <div className="flex items-center justify-center py-8">
          <Loader2 className="w-8 h-8 animate-spin text-blue-500" />
        </div>
      )}
    </div>
  );
}
