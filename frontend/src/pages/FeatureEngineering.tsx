import { useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Label } from '@/components/ui/label';
import { Badge } from '@/components/ui/badge';
import { Switch } from '@/components/ui/switch';
import { Slider } from '@/components/ui/slider';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { dataApi, featuresApi } from '@/lib/api';
import type { FeatureSummary } from '@/lib/api';
import {
  Sparkles,
  CheckCircle2,
  AlertCircle,
  Loader2,
  Wand2,
  Settings,
  Layers,
  ArrowRight,
} from 'lucide-react';
import { Link } from 'react-router-dom';

export function FeatureEngineering() {
  const queryClient = useQueryClient();

  // Feature generation options
  const [includeNumerical, setIncludeNumerical] = useState(true);
  const [includeInteractions, setIncludeInteractions] = useState(true);
  const [includeBinning, setIncludeBinning] = useState(true);
  const [includeDatetime, setIncludeDatetime] = useState(true);

  // Preprocessing options
  const [imputationStrategy, setImputationStrategy] = useState('median');
  const [handleOutliers, setHandleOutliers] = useState(false);
  const [outlierStrategy, setOutlierStrategy] = useState('iqr_cap');
  const [outlierThreshold, setOutlierThreshold] = useState([1.5]);
  const [applyScaling, setApplyScaling] = useState(false);
  const [scalingStrategy, setScalingStrategy] = useState('standard');
  const [groupRare, setGroupRare] = useState(false);
  const [rareThreshold, setRareThreshold] = useState([1]);

  const { data: stateResponse } = useQuery({
    queryKey: ['pipelineState'],
    queryFn: async () => {
      const response = await dataApi.getState();
      return response.data;
    },
  });

  const state = stateResponse?.data;

  const generateMutation = useMutation({
    mutationFn: () =>
      featuresApi.generate({
        include_numerical: includeNumerical,
        include_interactions: includeInteractions,
        include_binning: includeBinning,
        include_datetime: includeDatetime,
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['pipelineState'] });
    },
  });

  const preprocessMutation = useMutation({
    mutationFn: () =>
      featuresApi.preprocess({
        imputation_strategy: imputationStrategy,
        handle_outliers: handleOutliers,
        outlier_strategy: handleOutliers ? outlierStrategy : undefined,
        outlier_threshold: outlierThreshold[0],
        apply_scaling: applyScaling,
        scaling_strategy: applyScaling ? scalingStrategy : undefined,
        group_rare: groupRare,
        rare_threshold: rareThreshold[0] / 100,
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['pipelineState'] });
    },
  });

  const featureSummary = generateMutation.data?.data?.data as FeatureSummary | undefined;

  if (!state?.has_problem) {
    return (
      <div className="space-y-6">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Feature Engineering</h1>
          <p className="text-muted-foreground mt-2">
            Generate and preprocess features for your ML model
          </p>
        </div>
        <Card className="border-yellow-500/50 bg-yellow-50/50 dark:bg-yellow-950/20">
          <CardContent className="pt-6">
            <div className="flex items-center gap-4">
              <AlertCircle className="w-8 h-8 text-yellow-600" />
              <div>
                <h3 className="font-semibold">Complete Previous Steps First</h3>
                <p className="text-muted-foreground">
                  Please load data and define your problem in the Data Overview page.
                </p>
                <Link to="/data">
                  <Button className="mt-2" variant="outline">
                    Go to Data Overview <ArrowRight className="w-4 h-4 ml-2" />
                  </Button>
                </Link>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Feature Engineering</h1>
        <p className="text-muted-foreground mt-2">
          Generate and preprocess features for your ML model
        </p>
      </div>

      {/* Feature Generation */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Wand2 className="w-5 h-5" />
            Feature Generation
          </CardTitle>
          <CardDescription>
            Automatically create new features from your data
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="flex items-center justify-between p-4 bg-slate-50 dark:bg-slate-900 rounded-lg">
              <div>
                <Label className="font-medium">Numerical Transformations</Label>
                <p className="text-sm text-muted-foreground">
                  log, sqrt, square, cube
                </p>
              </div>
              <Switch
                checked={includeNumerical}
                onCheckedChange={setIncludeNumerical}
              />
            </div>

            <div className="flex items-center justify-between p-4 bg-slate-50 dark:bg-slate-900 rounded-lg">
              <div>
                <Label className="font-medium">Feature Interactions</Label>
                <p className="text-sm text-muted-foreground">
                  multiply, divide, add, subtract
                </p>
              </div>
              <Switch
                checked={includeInteractions}
                onCheckedChange={setIncludeInteractions}
              />
            </div>

            <div className="flex items-center justify-between p-4 bg-slate-50 dark:bg-slate-900 rounded-lg">
              <div>
                <Label className="font-medium">Binning</Label>
                <p className="text-sm text-muted-foreground">
                  Discretize continuous features
                </p>
              </div>
              <Switch
                checked={includeBinning}
                onCheckedChange={setIncludeBinning}
              />
            </div>

            <div className="flex items-center justify-between p-4 bg-slate-50 dark:bg-slate-900 rounded-lg">
              <div>
                <Label className="font-medium">Datetime Features</Label>
                <p className="text-sm text-muted-foreground">
                  year, month, day, hour
                </p>
              </div>
              <Switch
                checked={includeDatetime}
                onCheckedChange={setIncludeDatetime}
              />
            </div>
          </div>

          <Button
            onClick={() => generateMutation.mutate()}
            disabled={generateMutation.isPending}
            className="w-full md:w-auto"
            size="lg"
          >
            {generateMutation.isPending ? (
              <Loader2 className="w-4 h-4 mr-2 animate-spin" />
            ) : (
              <Sparkles className="w-4 h-4 mr-2" />
            )}
            Generate Features
          </Button>

          {generateMutation.isSuccess && featureSummary && (
            <div className="space-y-4">
              <div className="flex items-center gap-2 text-green-600">
                <CheckCircle2 className="w-5 h-5" />
                <span>
                  Generated {featureSummary.generated_features} new features!
                </span>
              </div>

              <div className="grid grid-cols-3 gap-4">
                <div className="text-center p-4 bg-blue-50 dark:bg-blue-950/30 rounded-lg">
                  <div className="text-2xl font-bold text-blue-600">
                    {featureSummary.original_features}
                  </div>
                  <div className="text-sm text-muted-foreground">Original</div>
                </div>
                <div className="text-center p-4 bg-green-50 dark:bg-green-950/30 rounded-lg">
                  <div className="text-2xl font-bold text-green-600">
                    +{featureSummary.generated_features}
                  </div>
                  <div className="text-sm text-muted-foreground">Generated</div>
                </div>
                <div className="text-center p-4 bg-purple-50 dark:bg-purple-950/30 rounded-lg">
                  <div className="text-2xl font-bold text-purple-600">
                    {featureSummary.total_features}
                  </div>
                  <div className="text-sm text-muted-foreground">Total</div>
                </div>
              </div>

              <Tabs defaultValue="transformations" className="mt-4">
                <TabsList>
                  <TabsTrigger value="transformations">
                    Transforms ({featureSummary.feature_categories?.transformations || 0})
                  </TabsTrigger>
                  <TabsTrigger value="interactions">
                    Interactions ({featureSummary.feature_categories?.interactions || 0})
                  </TabsTrigger>
                  <TabsTrigger value="binned">
                    Binned ({featureSummary.feature_categories?.binned || 0})
                  </TabsTrigger>
                </TabsList>

                <TabsContent value="transformations" className="mt-4">
                  <div className="flex flex-wrap gap-2">
                    {featureSummary.sample_features?.transformations?.map((f) => (
                      <Badge key={f} variant="outline">
                        {f}
                      </Badge>
                    ))}
                    {(!featureSummary.sample_features?.transformations ||
                      featureSummary.sample_features.transformations.length === 0) && (
                      <p className="text-muted-foreground">No transformation features</p>
                    )}
                  </div>
                </TabsContent>

                <TabsContent value="interactions" className="mt-4">
                  <div className="flex flex-wrap gap-2">
                    {featureSummary.sample_features?.interactions?.map((f) => (
                      <Badge key={f} variant="outline">
                        {f}
                      </Badge>
                    ))}
                  </div>
                </TabsContent>

                <TabsContent value="binned" className="mt-4">
                  <div className="flex flex-wrap gap-2">
                    {featureSummary.sample_features?.binned?.map((f) => (
                      <Badge key={f} variant="outline">
                        {f}
                      </Badge>
                    ))}
                  </div>
                </TabsContent>
              </Tabs>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Preprocessing */}
      {state?.has_features && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Settings className="w-5 h-5" />
              Feature Preprocessing
            </CardTitle>
            <CardDescription>
              Handle missing values, outliers, and encode categorical variables
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Imputation */}
              <div className="space-y-3">
                <Label className="font-medium">Missing Value Imputation</Label>
                <Select value={imputationStrategy} onValueChange={setImputationStrategy}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="median">Median</SelectItem>
                    <SelectItem value="mean">Mean</SelectItem>
                    <SelectItem value="mode">Mode</SelectItem>
                    <SelectItem value="drop">Drop rows</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {/* Scaling */}
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <Label className="font-medium">Feature Scaling</Label>
                  <Switch checked={applyScaling} onCheckedChange={setApplyScaling} />
                </div>
                {applyScaling && (
                  <Select value={scalingStrategy} onValueChange={setScalingStrategy}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="standard">Standard (Z-score)</SelectItem>
                      <SelectItem value="minmax">MinMax [0,1]</SelectItem>
                      <SelectItem value="robust">Robust (Median/IQR)</SelectItem>
                    </SelectContent>
                  </Select>
                )}
              </div>

              {/* Outliers */}
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <Label className="font-medium">Handle Outliers</Label>
                  <Switch checked={handleOutliers} onCheckedChange={setHandleOutliers} />
                </div>
                {handleOutliers && (
                  <>
                    <Select value={outlierStrategy} onValueChange={setOutlierStrategy}>
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="iqr_cap">IQR Cap (Winsorize)</SelectItem>
                        <SelectItem value="zscore_cap">Z-score Cap</SelectItem>
                        <SelectItem value="iqr_remove">IQR Remove</SelectItem>
                        <SelectItem value="zscore_remove">Z-score Remove</SelectItem>
                      </SelectContent>
                    </Select>
                    <div>
                      <Label className="text-sm text-muted-foreground">
                        Threshold: {outlierThreshold[0]}
                      </Label>
                      <Slider
                        value={outlierThreshold}
                        onValueChange={setOutlierThreshold}
                        min={1}
                        max={3}
                        step={0.1}
                      />
                    </div>
                  </>
                )}
              </div>

              {/* Rare Categories */}
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <Label className="font-medium">Group Rare Categories</Label>
                  <Switch checked={groupRare} onCheckedChange={setGroupRare} />
                </div>
                {groupRare && (
                  <div>
                    <Label className="text-sm text-muted-foreground">
                      Threshold: {rareThreshold[0]}%
                    </Label>
                    <Slider
                      value={rareThreshold}
                      onValueChange={setRareThreshold}
                      min={0.1}
                      max={5}
                      step={0.1}
                    />
                  </div>
                )}
              </div>
            </div>

            <Button
              onClick={() => preprocessMutation.mutate()}
              disabled={preprocessMutation.isPending}
              className="w-full md:w-auto"
              size="lg"
            >
              {preprocessMutation.isPending ? (
                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
              ) : (
                <Layers className="w-4 h-4 mr-2" />
              )}
              Preprocess Features
            </Button>

            {preprocessMutation.isSuccess && (
              <div className="flex items-center gap-2 text-green-600">
                <CheckCircle2 className="w-5 h-5" />
                <span>Features preprocessed successfully!</span>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* Next Steps */}
      {state?.has_preprocessed && (
        <Card className="border-green-500/50 bg-green-50/50 dark:bg-green-950/20">
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-4">
                <CheckCircle2 className="w-8 h-8 text-green-600" />
                <div>
                  <h3 className="font-semibold">Ready for Training!</h3>
                  <p className="text-muted-foreground">
                    Your features are preprocessed and ready for model training.
                  </p>
                </div>
              </div>
              <Link to="/training">
                <Button>
                  Train Model <ArrowRight className="w-4 h-4 ml-2" />
                </Button>
              </Link>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
