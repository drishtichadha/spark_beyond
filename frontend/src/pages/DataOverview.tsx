import { useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { dataApi } from '@/lib/api';
import type { QualityReport, SchemaInfo, ProfileReport } from '@/lib/api';
import {
  Database,
  Upload,
  CheckCircle2,
  AlertCircle,
  AlertTriangle,
  Loader2,
  FileText,
  BarChart2,
  Search,
  PieChart,
  TrendingUp,
} from 'lucide-react';
import { Progress } from '@/components/ui/progress';

export function DataOverview() {
  const queryClient = useQueryClient();
  const [filePath, setFilePath] = useState('data/bank-additional-full.csv');
  const [target, setTarget] = useState('y');
  const [problemType, setProblemType] = useState('classification');
  const [desiredResult, setDesiredResult] = useState('yes');
  const [dateColumn, setDateColumn] = useState<string | undefined>(undefined);

  const { data: stateResponse } = useQuery({
    queryKey: ['pipelineState'],
    queryFn: async () => {
      const response = await dataApi.getState();
      return response.data;
    },
  });

  const state = stateResponse?.data;

  const loadDataMutation = useMutation({
    mutationFn: (path: string) => dataApi.loadData(path),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['pipelineState'] });
    },
  });

  const setProblemMutation = useMutation({
    mutationFn: () => dataApi.setProblem(
      target,
      problemType,
      problemType === 'classification' ? desiredResult : undefined,
      dateColumn
    ),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['pipelineState'] });
      queryClient.invalidateQueries({ queryKey: ['schema'] });
    },
  });

  const qualityCheckMutation = useMutation({
    mutationFn: () => dataApi.runQualityCheck(),
  });

  const profileMutation = useMutation({
    mutationFn: () => dataApi.runProfile(true, 50000),
  });

  const schemaData = setProblemMutation.data?.data?.data as SchemaInfo | undefined;
  const qualityData = qualityCheckMutation.data?.data?.data as QualityReport | undefined;
  const profileData = profileMutation.data?.data?.data as ProfileReport | undefined;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Data Overview</h1>
        <p className="text-muted-foreground mt-2">
          Load your dataset, validate schema, and check data quality
        </p>
      </div>

      {/* Load Data Section */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Database className="w-5 h-5" />
            Load Dataset
          </CardTitle>
          <CardDescription>
            Enter the path to your CSV file or use the default dataset
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex gap-4">
            <div className="flex-1">
              <Label htmlFor="filePath">File Path</Label>
              <Input
                id="filePath"
                value={filePath}
                onChange={(e) => setFilePath(e.target.value)}
                placeholder="path/to/your/data.csv"
              />
            </div>
            <div className="flex items-end">
              <Button
                onClick={() => loadDataMutation.mutate(filePath)}
                disabled={loadDataMutation.isPending}
              >
                {loadDataMutation.isPending ? (
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                ) : (
                  <Upload className="w-4 h-4 mr-2" />
                )}
                Load Data
              </Button>
            </div>
          </div>

          {loadDataMutation.isSuccess && (
            <div className="flex items-center gap-2 text-green-600">
              <CheckCircle2 className="w-5 h-5" />
              <span>{loadDataMutation.data.data.message}</span>
            </div>
          )}

          {loadDataMutation.isError && (
            <div className="flex items-center gap-2 text-red-600">
              <AlertCircle className="w-5 h-5" />
              <span>Failed to load data</span>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Dataset Info */}
      {state?.dataset_info && (
        <Card>
          <CardHeader>
            <CardTitle>Dataset Information</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="text-center p-4 bg-blue-50 dark:bg-blue-950/30 rounded-lg border border-blue-200 dark:border-blue-800">
                <div className="text-3xl font-bold text-blue-600">
                  {state.dataset_info.rows.toLocaleString()}
                </div>
                <div className="text-sm text-muted-foreground">Rows</div>
              </div>
              <div className="text-center p-4 bg-purple-50 dark:bg-purple-950/30 rounded-lg border border-purple-200 dark:border-purple-800">
                <div className="text-3xl font-bold text-purple-600">
                  {state.dataset_info.columns}
                </div>
                <div className="text-sm text-muted-foreground">Columns</div>
              </div>
            </div>

            <div className="mt-6">
              <h4 className="font-semibold mb-2">Columns</h4>
              <div className="flex flex-wrap gap-2">
                {state.dataset_info.column_names.map((col) => (
                  <Badge key={col} variant="outline">
                    {col}
                    <span className="ml-1 text-xs text-muted-foreground">
                      ({state.dataset_info?.column_types[col]})
                    </span>
                  </Badge>
                ))}
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Problem Definition */}
      {state?.has_data && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <FileText className="w-5 h-5" />
              Problem Definition
            </CardTitle>
            <CardDescription>
              Define your ML problem - target column and problem type
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div>
                <Label htmlFor="target">Target Column</Label>
                <Select value={target} onValueChange={setTarget}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select target" />
                  </SelectTrigger>
                  <SelectContent>
                    {state.dataset_info?.column_names.map((col) => (
                      <SelectItem key={col} value={col}>
                        {col}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div>
                <Label htmlFor="problemType">Problem Type</Label>
                <Select value={problemType} onValueChange={setProblemType}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select type" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="classification">Classification</SelectItem>
                    <SelectItem value="regression">Regression</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {problemType === 'classification' && (
                <div>
                  <Label htmlFor="desiredResult">Desired Result</Label>
                  <Input
                    id="desiredResult"
                    value={desiredResult}
                    onChange={(e) => setDesiredResult(e.target.value)}
                    placeholder="e.g., yes, 1, True"
                  />
                </div>
              )}

              <div>
                <Label htmlFor="dateColumn">Date Column (Optional)</Label>
                <Select value={dateColumn ?? "none"} onValueChange={(val) => setDateColumn(val === "none" ? undefined : val)}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select Date Column" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="none">None</SelectItem>
                    {state.dataset_info?.column_names.map((col) => (
                      <SelectItem key={col} value={col}>
                        {col}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

            </div>

            <Button
              onClick={() => setProblemMutation.mutate()}
              disabled={setProblemMutation.isPending}
            >
              {setProblemMutation.isPending ? (
                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
              ) : (
                <CheckCircle2 className="w-4 h-4 mr-2" />
              )}
              Validate Schema
            </Button>

            {setProblemMutation.isSuccess && (
              <div className="flex items-center gap-2 text-green-600">
                <CheckCircle2 className="w-5 h-5" />
                <span>Schema validated successfully!</span>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* Schema Information */}
      {schemaData && (
        <Card>
          <CardHeader>
            <CardTitle>Schema Information</CardTitle>
          </CardHeader>
          <CardContent>
            <Tabs defaultValue="categorical">
              <TabsList>
                <TabsTrigger value="categorical">
                  Categorical ({schemaData.categorical?.length || 0})
                </TabsTrigger>
                <TabsTrigger value="numerical">
                  Numerical ({schemaData.numerical?.length || 0})
                </TabsTrigger>
                <TabsTrigger value="boolean">
                  Boolean ({schemaData.boolean?.length || 0})
                </TabsTrigger>
              </TabsList>

              <TabsContent value="categorical" className="mt-4">
                <div className="grid gap-4 md:grid-cols-2">
                  {schemaData.categorical?.map((col) => (
                    <Card key={col.name} className="bg-slate-50 dark:bg-slate-900">
                      <CardHeader className="py-3">
                        <CardTitle className="text-base">{col.name}</CardTitle>
                      </CardHeader>
                      <CardContent className="py-3">
                        <div className="grid grid-cols-2 gap-2 text-sm">
                          <div>
                            <span className="text-muted-foreground">Distinct: </span>
                            <span className="font-medium">{col.distinct_count}</span>
                          </div>
                          <div>
                            <span className="text-muted-foreground">Nulls: </span>
                            <span className="font-medium">{col.null_count}</span>
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  ))}
                </div>
              </TabsContent>

              <TabsContent value="numerical" className="mt-4">
                <div className="grid gap-4 md:grid-cols-2">
                  {schemaData.numerical?.map((col) => (
                    <Card key={col.name} className="bg-slate-50 dark:bg-slate-900">
                      <CardHeader className="py-3">
                        <CardTitle className="text-base">{col.name}</CardTitle>
                      </CardHeader>
                      <CardContent className="py-3">
                        <div className="grid grid-cols-2 gap-2 text-sm">
                          <div>
                            <span className="text-muted-foreground">Mean: </span>
                            <span className="font-medium">{col.mean?.toFixed(2)}</span>
                          </div>
                          <div>
                            <span className="text-muted-foreground">Std: </span>
                            <span className="font-medium">{col.std?.toFixed(2)}</span>
                          </div>
                          <div>
                            <span className="text-muted-foreground">Min: </span>
                            <span className="font-medium">{col.min?.toFixed(2)}</span>
                          </div>
                          <div>
                            <span className="text-muted-foreground">Max: </span>
                            <span className="font-medium">{col.max?.toFixed(2)}</span>
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  ))}
                </div>
              </TabsContent>

              <TabsContent value="boolean" className="mt-4">
                <div className="flex flex-wrap gap-2">
                  {schemaData.boolean?.map((col) => (
                    <Badge key={col.name} variant="outline">
                      {col.name}
                    </Badge>
                  ))}
                  {(!schemaData.boolean || schemaData.boolean.length === 0) && (
                    <p className="text-muted-foreground">No boolean columns found</p>
                  )}
                </div>
              </TabsContent>
            </Tabs>
          </CardContent>
        </Card>
      )}

      {/* Data Quality Check */}
      {state?.has_problem && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <BarChart2 className="w-5 h-5" />
              Data Quality Check
            </CardTitle>
            <CardDescription>
              Run automated quality checks on your dataset
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <Button
              onClick={() => qualityCheckMutation.mutate()}
              disabled={qualityCheckMutation.isPending}
              variant="outline"
            >
              {qualityCheckMutation.isPending ? (
                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
              ) : (
                <BarChart2 className="w-4 h-4 mr-2" />
              )}
              Run Quality Check
            </Button>

            {qualityCheckMutation.isError && (
              <div className="flex items-center gap-2 p-3 bg-red-50 dark:bg-red-950/30 rounded-lg border border-red-200 dark:border-red-800">
                <AlertCircle className="w-5 h-5 text-red-500" />
                <span className="text-sm text-red-700 dark:text-red-300">
                  Failed to run quality check: {
                    (qualityCheckMutation.error as any)?.response?.data?.detail ||
                    (qualityCheckMutation.error as Error)?.message ||
                    'Unknown error'
                  }
                </span>
              </div>
            )}

            {qualityData && (
              <div className="space-y-4">
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="text-center p-4 bg-green-50 dark:bg-green-950/30 rounded-lg border border-green-200 dark:border-green-800">
                    <div className="text-3xl font-bold text-green-600">
                      {qualityData.quality_score}
                    </div>
                    <div className="text-sm text-muted-foreground">Quality Score</div>
                  </div>
                  <div className="text-center p-4 bg-slate-50 dark:bg-slate-900 rounded-lg">
                    <div className="text-2xl font-bold">
                      {qualityData.row_count.toLocaleString()}
                    </div>
                    <div className="text-sm text-muted-foreground">Rows</div>
                  </div>
                  <div className="text-center p-4 bg-slate-50 dark:bg-slate-900 rounded-lg">
                    <div className="text-2xl font-bold">{qualityData.column_count}</div>
                    <div className="text-sm text-muted-foreground">Columns</div>
                  </div>
                  <div className="text-center p-4 bg-slate-50 dark:bg-slate-900 rounded-lg">
                    <div className="text-2xl font-bold">{qualityData.duplicate_count}</div>
                    <div className="text-sm text-muted-foreground">Duplicates</div>
                  </div>
                </div>

                {qualityData.issues && qualityData.issues.length > 0 && (
                  <div>
                    <h4 className="font-semibold mb-2 flex items-center gap-2">
                      <AlertTriangle className="w-4 h-4 text-yellow-500" />
                      Issues Found
                    </h4>
                    <div className="space-y-2">
                      {qualityData.issues.slice(0, 5).map((issue, i) => (
                        <div
                          key={i}
                          className="flex items-center gap-2 p-2 bg-yellow-50 dark:bg-yellow-950/30 rounded border border-yellow-200 dark:border-yellow-800"
                        >
                          <Badge
                            variant={
                              issue.severity === 'high'
                                ? 'destructive'
                                : issue.severity === 'medium'
                                ? 'warning'
                                : 'secondary'
                            }
                          >
                            {issue.severity}
                          </Badge>
                          <span className="font-medium">{issue.column}:</span>
                          <span className="text-muted-foreground">{issue.issue}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {qualityData.recommendations && qualityData.recommendations.length > 0 && (
                  <div>
                    <h4 className="font-semibold mb-2">Recommendations</h4>
                    <div className="space-y-2">
                      {qualityData.recommendations.slice(0, 5).map((rec, i) => (
                        <div
                          key={i}
                          className="flex items-center gap-2 p-2 bg-blue-50 dark:bg-blue-950/30 rounded border border-blue-200 dark:border-blue-800"
                        >
                          <Badge variant="outline">{rec.priority}</Badge>
                          <span className="font-medium">{rec.column}:</span>
                          <span className="text-muted-foreground">{rec.action}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* Data Profiler */}
      {state?.has_data && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Search className="w-5 h-5" />
              Data Profiler
            </CardTitle>
            <CardDescription>
              Run comprehensive data profiling to discover patterns, distributions, and correlations
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <Button
              onClick={() => profileMutation.mutate()}
              disabled={profileMutation.isPending}
              variant="outline"
            >
              {profileMutation.isPending ? (
                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
              ) : (
                <PieChart className="w-4 h-4 mr-2" />
              )}
              Run Data Profile
            </Button>

            {profileMutation.isPending && (
              <div className="space-y-2">
                <p className="text-sm text-muted-foreground">
                  Analyzing data distributions and patterns...
                </p>
                <Progress value={undefined} className="h-2" />
              </div>
            )}

            {profileMutation.isError && (
              <div className="flex items-center gap-2 p-3 bg-red-50 dark:bg-red-950/30 rounded-lg border border-red-200 dark:border-red-800">
                <AlertCircle className="w-5 h-5 text-red-500" />
                <span className="text-sm text-red-700 dark:text-red-300">
                  Failed to run data profile: {
                    (profileMutation.error as any)?.response?.data?.detail ||
                    (profileMutation.error as Error)?.message ||
                    'Unknown error'
                  }
                </span>
              </div>
            )}

            {profileData && (
              <div className="space-y-6">
                {/* Summary Stats */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="text-center p-4 bg-blue-50 dark:bg-blue-950/30 rounded-lg border border-blue-200 dark:border-blue-800">
                    <div className="text-2xl font-bold text-blue-600">
                      {profileData.summary?.n_rows?.toLocaleString() || 0}
                    </div>
                    <div className="text-sm text-muted-foreground">Total Rows</div>
                  </div>
                  <div className="text-center p-4 bg-purple-50 dark:bg-purple-950/30 rounded-lg border border-purple-200 dark:border-purple-800">
                    <div className="text-2xl font-bold text-purple-600">
                      {profileData.summary?.n_columns || 0}
                    </div>
                    <div className="text-sm text-muted-foreground">Columns</div>
                  </div>
                  <div className="text-center p-4 bg-orange-50 dark:bg-orange-950/30 rounded-lg border border-orange-200 dark:border-orange-800">
                    <div className="text-2xl font-bold text-orange-600">
                      {profileData.summary?.missing_cells_pct?.toFixed(1) || 0}%
                    </div>
                    <div className="text-sm text-muted-foreground">Missing</div>
                  </div>
                  <div className="text-center p-4 bg-green-50 dark:bg-green-950/30 rounded-lg border border-green-200 dark:border-green-800">
                    <div className="text-2xl font-bold text-green-600">
                      {Object.keys(profileData.summary?.types || {}).length}
                    </div>
                    <div className="text-sm text-muted-foreground">Data Types</div>
                  </div>
                </div>

                {/* Sample Info */}
                {profileData.sample_info?.sampled && (
                  <div className="p-3 bg-yellow-50 dark:bg-yellow-950/30 rounded-lg border border-yellow-200 dark:border-yellow-800">
                    <div className="flex items-center gap-2">
                      <AlertTriangle className="w-4 h-4 text-yellow-600" />
                      <span className="text-sm">
                        Profiled on a sample of {profileData.sample_info.sampled_rows?.toLocaleString()} rows
                        ({((profileData.sample_info.sample_ratio || 0) * 100).toFixed(1)}% of data)
                      </span>
                    </div>
                  </div>
                )}

                {/* Data Types Distribution */}
                {profileData.summary?.types && Object.keys(profileData.summary.types).length > 0 && (
                  <div>
                    <h4 className="font-semibold mb-3">Column Types Distribution</h4>
                    <div className="flex flex-wrap gap-2">
                      {Object.entries(profileData.summary.types).map(([type, count]) => (
                        <Badge key={type} variant="outline" className="text-sm py-1 px-3">
                          {type}: {count}
                        </Badge>
                      ))}
                    </div>
                  </div>
                )}

                {/* Missing Values */}
                {profileData.missing_values && Object.keys(profileData.missing_values).length > 0 && (
                  <div>
                    <h4 className="font-semibold mb-3 flex items-center gap-2">
                      <AlertTriangle className="w-4 h-4 text-orange-500" />
                      Columns with Missing Values
                    </h4>
                    <div className="space-y-2">
                      {Object.entries(profileData.missing_values)
                        .sort(([, a], [, b]) => b - a)
                        .slice(0, 10)
                        .map(([col, pct]) => (
                          <div key={col} className="flex items-center gap-3">
                            <span className="w-40 truncate font-medium text-sm">{col}</span>
                            <div className="flex-1">
                              <Progress value={pct} className="h-2" />
                            </div>
                            <span className="w-16 text-right text-sm text-muted-foreground">
                              {pct.toFixed(1)}%
                            </span>
                          </div>
                        ))}
                    </div>
                  </div>
                )}

                {/* Alerts */}
                {profileData.alerts && profileData.alerts.length > 0 && (
                  <div>
                    <h4 className="font-semibold mb-3 flex items-center gap-2">
                      <AlertCircle className="w-4 h-4 text-red-500" />
                      Data Quality Alerts ({profileData.alerts.length})
                    </h4>
                    <div className="space-y-2 max-h-60 overflow-y-auto">
                      {profileData.alerts.slice(0, 10).map((alert, i) => (
                        <div
                          key={i}
                          className="flex items-start gap-2 p-2 bg-red-50 dark:bg-red-950/30 rounded border border-red-200 dark:border-red-800"
                        >
                          <Badge variant="destructive" className="text-xs">
                            {alert.type}
                          </Badge>
                          <div className="flex-1">
                            <span className="font-medium text-sm">{alert.column}: </span>
                            <span className="text-sm text-muted-foreground">{alert.message}</span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Recommendations */}
                {profileData.recommendations && profileData.recommendations.length > 0 && (
                  <div>
                    <h4 className="font-semibold mb-3 flex items-center gap-2">
                      <TrendingUp className="w-4 h-4 text-blue-500" />
                      Preprocessing Recommendations
                    </h4>
                    <div className="space-y-2">
                      {profileData.recommendations.slice(0, 8).map((rec, i) => (
                        <div
                          key={i}
                          className="flex items-start gap-2 p-3 bg-blue-50 dark:bg-blue-950/30 rounded border border-blue-200 dark:border-blue-800"
                        >
                          <Badge
                            variant={
                              rec.priority === 'high'
                                ? 'destructive'
                                : rec.priority === 'medium'
                                ? 'warning'
                                : 'secondary'
                            }
                            className="text-xs"
                          >
                            {rec.priority}
                          </Badge>
                          <div className="flex-1">
                            <div className="font-medium text-sm">{rec.column}</div>
                            <div className="text-sm text-muted-foreground">{rec.issue}</div>
                            <div className="text-sm text-blue-600 dark:text-blue-400 mt-1">
                              → {rec.action}
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  );
}
