import { useState, useEffect, useRef, useCallback } from 'react';
import { useQuery, useInfiniteQuery } from '@tanstack/react-query';
import { Link } from 'react-router-dom';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Slider } from '@/components/ui/slider';
import { Label } from '@/components/ui/label';
import { Progress } from '@/components/ui/progress';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { dataApi, insightsApi } from '@/lib/api';
import { useDebouncedValue } from '@/lib/hooks';
import { useMicrosegmentStream } from '@/lib/useMicrosegmentStream';
import type { FeatureImportance, InsightAnalysis, MicrosegmentsPaginatedResponse } from '@/lib/api';
import {
  AlertCircle,
  ArrowRight,
  Loader2,
  BarChart2,
  TrendingUp,
  Layers,
  ArrowUp,
  ArrowDown,
  Play,
  Square,
  RefreshCw,
  CheckCircle2,
  XCircle,
  Zap,
} from 'lucide-react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ScatterChart,
  Scatter,
  ZAxis,
  Cell,
} from 'recharts';

export function Insights() {
  const [topN, setTopN] = useState([20]);
  const [minLift, setMinLift] = useState([1.1]);
  const [minSupport, setMinSupport] = useState([0.01]);
  // Microsegment parameters
  const [maxDepth, setMaxDepth] = useState([3]);
  const [topNFeatures, setTopNFeatures] = useState([50]);
  const [maxMicrosegments, setMaxMicrosegments] = useState([100]);
  // Microsegment sorting and pagination
  const [sortBy, setSortBy] = useState<'lift' | 'support' | 'rig' | 'support_count'>('lift');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc');
  const loadMoreRef = useRef<HTMLDivElement>(null);
  // Streaming mode toggle
  const [useStreaming, setUseStreaming] = useState(true);
  // Track active tab to only fetch data when tab is visited
  const [activeTab, setActiveTab] = useState('importance');

  // WebSocket streaming hook
  const {
    state: streamState,
    startDiscovery,
    cancelDiscovery,
    resetState: resetStreamState,
  } = useMicrosegmentStream();

  // Debounce slider values to avoid triggering API calls on every slider movement
  // Only the debounced values are used for API calls
  const debouncedMinLift = useDebouncedValue(minLift[0], 300);
  const debouncedMinSupport = useDebouncedValue(minSupport[0], 300);
  const debouncedTopN = useDebouncedValue(topN[0], 300);
  // Microsegment params use longer debounce since they trigger recomputation
  const debouncedMaxDepth = useDebouncedValue(maxDepth[0], 500);
  const debouncedTopNFeatures = useDebouncedValue(topNFeatures[0], 500);
  const debouncedMaxMicrosegments = useDebouncedValue(maxMicrosegments[0], 500);

  const { data: stateResponse } = useQuery({
    queryKey: ['pipelineState'],
    queryFn: async () => {
      const response = await dataApi.getState();
      return response.data;
    },
  });

  const state = stateResponse?.data;

  const { data: importanceResponse, isLoading: importanceLoading } = useQuery({
    queryKey: ['featureImportance', debouncedTopN],
    queryFn: async () => {
      const response = await insightsApi.getFeatureImportance(debouncedTopN);
      return response.data;
    },
    enabled: state?.has_model,
  });

  const { data: liftSupportResponse, isLoading: liftLoading, isFetching: liftFetching, refetch: refetchLift } = useQuery({
    queryKey: ['liftSupport', debouncedMinSupport, debouncedMinLift, debouncedMaxDepth, debouncedTopNFeatures, debouncedMaxMicrosegments],
    queryFn: async () => {
      const response = await insightsApi.getLiftSupport(
        debouncedMinSupport,
        debouncedMinLift,
        debouncedMaxDepth,
        debouncedTopNFeatures,
        debouncedMaxMicrosegments
      );
      return response.data;
    },
    enabled: state?.has_model && activeTab === 'lift',
    staleTime: 1000 * 60 * 5, // Cache for 5 minutes since backend now caches
    refetchOnMount: false, // Don't refetch when component remounts
  });

  // Infinite query for paginated microsegments with lazy loading
  const {
    data: microsegmentsData,
    isLoading: microsegmentsLoading,
    isFetchingNextPage,
    hasNextPage,
    fetchNextPage,
    refetch: refetchMicrosegments,
  } = useInfiniteQuery({
    queryKey: ['microsegmentsPaginated', debouncedMinSupport, debouncedMinLift, sortBy, sortOrder],
    queryFn: async ({ pageParam = 1 }) => {
      const response = await insightsApi.getMicrosegmentsPaginated({
        minSupport: debouncedMinSupport,
        minLift: debouncedMinLift,
        page: pageParam,
        pageSize: 20,
        sortBy,
        sortOrder,
      });
      return response.data.data as MicrosegmentsPaginatedResponse;
    },
    getNextPageParam: (lastPage) => {
      if (lastPage?.pagination?.has_next) {
        return lastPage.pagination.page + 1;
      }
      return undefined;
    },
    initialPageParam: 1,
    enabled: state?.has_model && activeTab === 'microsegments' && !useStreaming,
    staleTime: 1000 * 60 * 5,
    refetchOnMount: false, // Don't refetch when component remounts
  });

  // Flatten all pages of microsegments into a single array
  const allMicrosegments = microsegmentsData?.pages?.flatMap(page => page?.microsegments || []) || [];
  const totalMicrosegments = microsegmentsData?.pages?.[0]?.pagination?.total || 0;

  // Intersection observer for infinite scroll
  const handleLoadMore = useCallback(() => {
    if (hasNextPage && !isFetchingNextPage) {
      fetchNextPage();
    }
  }, [hasNextPage, isFetchingNextPage, fetchNextPage]);

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries[0].isIntersecting) {
          handleLoadMore();
        }
      },
      { threshold: 0.1 }
    );

    if (loadMoreRef.current) {
      observer.observe(loadMoreRef.current);
    }

    return () => observer.disconnect();
  }, [handleLoadMore]);

  const importance = importanceResponse?.data as FeatureImportance[] | undefined;
  const liftSupport = liftSupportResponse?.data as InsightAnalysis | undefined;

  if (!state?.has_model) {
    return (
      <div className="space-y-6">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Insights</h1>
          <p className="text-muted-foreground mt-2">
            Analyze feature importance and discover insights
          </p>
        </div>
        <Card className="border-yellow-500/50 bg-yellow-50/50 dark:bg-yellow-950/20">
          <CardContent className="pt-6">
            <div className="flex items-center gap-4">
              <AlertCircle className="w-8 h-8 text-yellow-600" />
              <div>
                <h3 className="font-semibold">Train Model First</h3>
                <p className="text-muted-foreground">
                  Please train your model to view insights.
                </p>
                <Link to="/training">
                  <Button className="mt-2" variant="outline">
                    Go to Training <ArrowRight className="w-4 h-4 ml-2" />
                  </Button>
                </Link>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  const importanceChartData = importance?.map((item) => ({
    name: item.feature.length > 25 ? item.feature.slice(0, 25) + '...' : item.feature,
    fullName: item.feature,
    importance: item.importance,
  })) || [];

  // Prepare scatter data for lift vs support
  const scatterData = liftSupport?.insights?.map((item) => ({
    x: item.Support_Value * 100,
    y: item.Lift_Value,
    z: item.RIG_Value * 100,
    name: item.Condition,
  })) || [];

  const COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6'];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Insights & Analysis</h1>
        <p className="text-muted-foreground mt-2">
          Explore feature importance, lift analysis, and model insights
        </p>
      </div>

      <Tabs defaultValue="importance" value={activeTab} onValueChange={setActiveTab} className="space-y-6">
        <TabsList className="grid w-full grid-cols-3 lg:w-auto lg:inline-flex">
          <TabsTrigger value="importance" className="flex items-center gap-2">
            <BarChart2 className="w-4 h-4" />
            Feature Importance
          </TabsTrigger>
          <TabsTrigger value="lift" className="flex items-center gap-2">
            <TrendingUp className="w-4 h-4" />
            Lift Analysis
          </TabsTrigger>
          <TabsTrigger value="microsegments" className="flex items-center gap-2">
            <Layers className="w-4 h-4" />
            Microsegments
          </TabsTrigger>
        </TabsList>

        {/* Feature Importance Tab */}
        <TabsContent value="importance" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <BarChart2 className="w-5 h-5" />
                Feature Importance
              </CardTitle>
              <CardDescription>
                XGBoost feature importance scores - higher values indicate more predictive features
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center gap-4">
                <Label>Top N Features:</Label>
                <Slider
                  value={topN}
                  onValueChange={setTopN}
                  min={10}
                  max={50}
                  step={5}
                  className="w-40"
                />
                <span className="text-sm text-muted-foreground">{topN[0]}</span>
              </div>

              {importanceLoading ? (
                <div className="flex items-center justify-center py-12">
                  <Loader2 className="w-8 h-8 animate-spin text-blue-500" />
                </div>
              ) : (
                <div className="h-[600px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                      data={importanceChartData}
                      layout="vertical"
                      margin={{ top: 5, right: 30, left: 150, bottom: 5 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis type="number" />
                      <YAxis
                        dataKey="name"
                        type="category"
                        tick={{ fontSize: 12 }}
                        width={140}
                      />
                      <Tooltip
                        formatter={(value, _name, props: any) => [
                          typeof value === 'number' ? value.toFixed(4) : value,
                          props.payload.fullName,
                        ]}
                      />
                      <Bar dataKey="importance" fill="#3b82f6" radius={[0, 4, 4, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Lift Analysis Tab */}
        <TabsContent value="lift" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <TrendingUp className="w-5 h-5" />
                Lift vs Support Analysis
              </CardTitle>
              <CardDescription>
                Discover feature conditions with high lift (predictive power) and support (coverage)
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex flex-wrap gap-6">
                <div className="flex items-center gap-4">
                  <Label>Min Lift:</Label>
                  <Slider
                    value={minLift}
                    onValueChange={setMinLift}
                    min={1.0}
                    max={5.0}
                    step={0.1}
                    className="w-32"
                  />
                  <span className="text-sm text-muted-foreground">x{minLift[0].toFixed(1)}</span>
                </div>
                <div className="flex items-center gap-4">
                  <Label>Min Support:</Label>
                  <Slider
                    value={minSupport}
                    onValueChange={setMinSupport}
                    min={0.005}
                    max={0.1}
                    step={0.005}
                    className="w-32"
                  />
                  <span className="text-sm text-muted-foreground">
                    {(minSupport[0] * 100).toFixed(1)}%
                  </span>
                </div>
                <Button variant="outline" onClick={() => refetchLift()}>
                  Refresh
                </Button>
              </div>

              {liftSupport && (
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 my-4">
                  <div className="text-center p-3 bg-blue-50 dark:bg-blue-950/30 rounded-lg">
                    <div className="text-xl font-bold text-blue-600">
                      {liftSupport.target_class}
                    </div>
                    <div className="text-xs text-muted-foreground">Target Class</div>
                  </div>
                  <div className="text-center p-3 bg-purple-50 dark:bg-purple-950/30 rounded-lg">
                    <div className="text-xl font-bold text-purple-600">
                      {(liftSupport.baseline_rate * 100).toFixed(1)}%
                    </div>
                    <div className="text-xs text-muted-foreground">Baseline Rate</div>
                  </div>
                  <div className="text-center p-3 bg-green-50 dark:bg-green-950/30 rounded-lg">
                    <div className="text-xl font-bold text-green-600">
                      {liftSupport.insights?.length || 0}
                    </div>
                    <div className="text-xs text-muted-foreground">Insights Found</div>
                  </div>
                  <div className="text-center p-3 bg-orange-50 dark:bg-orange-950/30 rounded-lg">
                    <div className="text-xl font-bold text-orange-600">
                      {liftSupport.total_count.toLocaleString()}
                    </div>
                    <div className="text-xs text-muted-foreground">Total Records</div>
                  </div>
                </div>
              )}

              {liftLoading && !liftSupport ? (
                <div className="flex items-center justify-center py-12">
                  <Loader2 className="w-8 h-8 animate-spin text-blue-500" />
                </div>
              ) : (
                <div className="relative">
                  {liftFetching && (
                    <div className="absolute top-2 right-2 flex items-center gap-2 text-sm text-muted-foreground bg-background/80 px-2 py-1 rounded">
                      <Loader2 className="w-4 h-4 animate-spin" />
                      Updating...
                    </div>
                  )}
                  {/* Scatter Plot */}
                  <div className="h-96">
                    <ResponsiveContainer width="100%" height="100%">
                      <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis
                          type="number"
                          dataKey="x"
                          name="Support"
                          unit="%"
                          label={{ value: 'Support (%)', position: 'bottom' }}
                        />
                        <YAxis
                          type="number"
                          dataKey="y"
                          name="Lift"
                          label={{ value: 'Lift', angle: -90, position: 'insideLeft' }}
                        />
                        <ZAxis type="number" dataKey="z" range={[50, 400]} />
                        <Tooltip
                          content={({ active, payload }) => {
                            if (active && payload && payload.length) {
                              const data = payload[0].payload;
                              return (
                                <div className="bg-white dark:bg-slate-800 p-3 rounded-lg shadow-lg border">
                                  <p className="font-medium text-sm">{data.name}</p>
                                  <p className="text-sm text-muted-foreground">
                                    Lift: x{data.y.toFixed(2)}
                                  </p>
                                  <p className="text-sm text-muted-foreground">
                                    Support: {data.x.toFixed(1)}%
                                  </p>
                                  <p className="text-sm text-muted-foreground">
                                    RIG: {(data.z / 100).toFixed(3)}
                                  </p>
                                </div>
                              );
                            }
                            return null;
                          }}
                        />
                        <Scatter name="Insights" data={scatterData}>
                          {scatterData.map((_entry, index) => (
                            <Cell
                              key={`cell-${index}`}
                              fill={COLORS[index % COLORS.length]}
                              fillOpacity={0.7}
                            />
                          ))}
                        </Scatter>
                      </ScatterChart>
                    </ResponsiveContainer>
                  </div>

                  {/* Insights Table */}
                  <div className="mt-6">
                    <h4 className="font-semibold mb-3">Top Insights</h4>
                    <div className="space-y-2 max-h-96 overflow-y-auto">
                      {liftSupport?.insights?.slice(0, 15).map((insight, i) => (
                        <div
                          key={i}
                          className="flex items-center justify-between p-3 bg-slate-50 dark:bg-slate-900 rounded-lg"
                        >
                          <div className="flex-1">
                            <p className="font-medium text-sm">{insight.Condition}</p>
                            <p className="text-xs text-muted-foreground">
                              Class Rate: {insight.Class_Rate}
                            </p>
                          </div>
                          <div className="flex items-center gap-3">
                            <Badge variant="default">{insight.Lift}</Badge>
                            <Badge variant="outline">{insight.Support}</Badge>
                            <Badge variant="secondary">RIG: {insight.RIG}</Badge>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Microsegments Tab */}
        <TabsContent value="microsegments" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Layers className="w-5 h-5" />
                Microsegments
                {useStreaming && (
                  <Badge variant="outline" className="ml-2 text-xs">
                    <Zap className="w-3 h-3 mr-1" />
                    Real-time
                  </Badge>
                )}
              </CardTitle>
              <CardDescription>
                Feature combinations that perform better than individual features
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* Microsegment Controls */}
              <div className="p-4 bg-slate-50 dark:bg-slate-900 rounded-lg space-y-4">
                <div className="flex items-center justify-between">
                  <h4 className="font-medium text-sm">Discovery Settings</h4>
                  <div className="flex items-center gap-2">
                    <Label className="text-xs text-muted-foreground">Streaming</Label>
                    <Button
                      variant={useStreaming ? 'default' : 'outline'}
                      size="sm"
                      className="h-7 px-2"
                      onClick={() => setUseStreaming(!useStreaming)}
                    >
                      {useStreaming ? (
                        <><Zap className="w-3 h-3 mr-1" /> On</>
                      ) : (
                        <><RefreshCw className="w-3 h-3 mr-1" /> Off</>
                      )}
                    </Button>
                  </div>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  <div className="space-y-2">
                    <Label className="text-sm">
                      Max Depth (conditions to combine)
                    </Label>
                    <div className="flex items-center gap-3">
                      <Slider
                        value={maxDepth}
                        onValueChange={setMaxDepth}
                        min={2}
                        max={5}
                        step={1}
                        className="flex-1"
                        disabled={streamState.status === 'discovering'}
                      />
                      <span className="text-sm font-medium w-8 text-center">
                        {maxDepth[0]}
                      </span>
                    </div>
                    <p className="text-xs text-muted-foreground">
                      Higher = deeper combinations (2-way, 3-way, etc.)
                    </p>
                  </div>
                  <div className="space-y-2">
                    <Label className="text-sm">
                      Top Features to Consider
                    </Label>
                    <div className="flex items-center gap-3">
                      <Slider
                        value={topNFeatures}
                        onValueChange={setTopNFeatures}
                        min={10}
                        max={100}
                        step={10}
                        className="flex-1"
                        disabled={streamState.status === 'discovering'}
                      />
                      <span className="text-sm font-medium w-8 text-center">
                        {topNFeatures[0]}
                      </span>
                    </div>
                    <p className="text-xs text-muted-foreground">
                      More features = more combinations explored
                    </p>
                  </div>
                  <div className="space-y-2">
                    <Label className="text-sm">
                      Max Microsegments
                    </Label>
                    <div className="flex items-center gap-3">
                      <Slider
                        value={maxMicrosegments}
                        onValueChange={setMaxMicrosegments}
                        min={10}
                        max={500}
                        step={10}
                        className="flex-1"
                        disabled={streamState.status === 'discovering'}
                      />
                      <span className="text-sm font-medium w-12 text-center">
                        {maxMicrosegments[0]}
                      </span>
                    </div>
                    <p className="text-xs text-muted-foreground">
                      Maximum results to return
                    </p>
                  </div>
                </div>
                <div className="flex items-center gap-4 pt-2">
                  {useStreaming ? (
                    <>
                      {streamState.status === 'idle' || streamState.status === 'complete' || streamState.status === 'error' ? (
                        <Button
                          variant="default"
                          size="sm"
                          onClick={() => {
                            resetStreamState();
                            startDiscovery({
                              minSupport: debouncedMinSupport,
                              minLift: debouncedMinLift,
                              maxDepth: maxDepth[0],
                              topNFeatures: topNFeatures[0],
                              maxMicrosegments: maxMicrosegments[0],
                            });
                          }}
                          className="flex items-center gap-2"
                        >
                          <Play className="w-4 h-4" />
                          Start Discovery
                        </Button>
                      ) : streamState.status === 'discovering' || streamState.status === 'connecting' ? (
                        <Button
                          variant="destructive"
                          size="sm"
                          onClick={cancelDiscovery}
                          className="flex items-center gap-2"
                        >
                          <Square className="w-4 h-4" />
                          Cancel
                        </Button>
                      ) : null}
                      {streamState.status === 'complete' && (
                        <span className="text-sm text-green-600 flex items-center gap-2">
                          <CheckCircle2 className="w-4 h-4" />
                          Discovery complete
                        </span>
                      )}
                      {streamState.status === 'error' && (
                        <span className="text-sm text-red-600 flex items-center gap-2">
                          <XCircle className="w-4 h-4" />
                          {streamState.error || 'An error occurred'}
                        </span>
                      )}
                    </>
                  ) : (
                    <>
                      <Button variant="outline" size="sm" onClick={() => refetchLift()}>
                        Refresh Analysis
                      </Button>
                      {liftFetching && (
                        <span className="text-sm text-muted-foreground flex items-center gap-2">
                          <Loader2 className="w-4 h-4 animate-spin" />
                          Discovering microsegments...
                        </span>
                      )}
                    </>
                  )}
                </div>
              </div>

              {/* Progress Indicator for Streaming */}
              {useStreaming && (streamState.status === 'discovering' || streamState.status === 'connecting') && (
                <div className="p-4 bg-blue-50 dark:bg-blue-950/30 rounded-lg space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium flex items-center gap-2">
                      <Loader2 className="w-4 h-4 animate-spin text-blue-600" />
                      {streamState.status === 'connecting' ? 'Connecting...' : 'Discovering microsegments...'}
                    </span>
                    <span className="text-sm text-blue-600 font-medium">
                      {streamState.progress.progress}%
                    </span>
                  </div>
                  <Progress value={streamState.progress.progress} className="h-2" />
                  <p className="text-xs text-muted-foreground">
                    {streamState.progress.message || 'Processing...'}
                  </p>
                  {streamState.microsegments.length > 0 && (
                    <p className="text-xs text-blue-600">
                      Found {streamState.microsegments.length} microsegments so far...
                    </p>
                  )}
                </div>
              )}

              {/* Sorting Controls */}
              <div className="flex flex-wrap items-center gap-4">
                <div className="flex items-center gap-2">
                  <Label className="text-sm whitespace-nowrap">Sort by:</Label>
                  <Select value={sortBy} onValueChange={(v) => setSortBy(v as typeof sortBy)}>
                    <SelectTrigger className="w-[140px]">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="lift">Lift</SelectItem>
                      <SelectItem value="support">Support</SelectItem>
                      <SelectItem value="rig">RIG</SelectItem>
                      <SelectItem value="support_count">Record Count</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setSortOrder(sortOrder === 'desc' ? 'asc' : 'desc')}
                  className="flex items-center gap-1"
                >
                  {sortOrder === 'desc' ? (
                    <>
                      <ArrowDown className="w-4 h-4" />
                      Descending
                    </>
                  ) : (
                    <>
                      <ArrowUp className="w-4 h-4" />
                      Ascending
                    </>
                  )}
                </Button>
                {!useStreaming && (
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => refetchMicrosegments()}
                    className="ml-auto"
                  >
                    Refresh
                  </Button>
                )}
              </div>

              {/* Microsegments Results - Streaming Mode */}
              {useStreaming ? (
                <>
                  {streamState.microsegments.length > 0 ? (
                    <div className="space-y-4">
                      <div className="flex items-center justify-between">
                        <span className="text-sm text-muted-foreground">
                          {streamState.status === 'complete' ? (
                            <>Showing {streamState.microsegments.length} microsegments</>
                          ) : (
                            <>Found {streamState.microsegments.length} microsegments (discovering...)</>
                          )}
                        </span>
                      </div>

                      {/* Sort microsegments based on current sort settings */}
                      {[...streamState.microsegments]
                        .sort((a, b) => {
                          const aVal = a[sortBy] ?? 0;
                          const bVal = b[sortBy] ?? 0;
                          return sortOrder === 'desc' ? bVal - aVal : aVal - bVal;
                        })
                        .map((micro, i) => (
                          <Card key={`stream-${micro.name}-${i}`} className="bg-slate-50 dark:bg-slate-900">
                            <CardContent className="pt-4">
                              <div className="flex items-start justify-between">
                                <div className="flex-1">
                                  <div className="flex items-center gap-2 mb-2">
                                    <h4 className="font-semibold text-sm">
                                      Microsegment #{i + 1}
                                    </h4>
                                    {micro.depth && (
                                      <Badge variant="secondary" className="text-xs">
                                        {micro.depth}-way
                                      </Badge>
                                    )}
                                  </div>
                                  <div className="flex flex-wrap gap-1">
                                    {micro.conditions.map((cond, j) => (
                                      <Badge key={j} variant="outline" className="text-xs">
                                        {cond}
                                      </Badge>
                                    ))}
                                  </div>
                                </div>
                                <div className="text-right space-y-1">
                                  <div className="text-2xl font-bold text-blue-600">
                                    x{micro.lift.toFixed(2)}
                                  </div>
                                  <div className="text-sm text-muted-foreground">
                                    {(micro.support * 100).toFixed(1)}% support
                                  </div>
                                  <div className="text-sm text-muted-foreground">
                                    {micro.support_count.toLocaleString()} records
                                  </div>
                                  <div className="text-xs text-muted-foreground">
                                    RIG: {micro.rig.toFixed(3)}
                                  </div>
                                </div>
                              </div>
                            </CardContent>
                          </Card>
                        ))}
                    </div>
                  ) : streamState.status === 'idle' ? (
                    <div className="text-center py-12 text-muted-foreground">
                      <Zap className="w-12 h-12 mx-auto mb-4 opacity-50" />
                      <p>Click "Start Discovery" to find microsegments in real-time.</p>
                      <p className="text-sm">Results will appear as they are discovered.</p>
                    </div>
                  ) : streamState.status === 'discovering' || streamState.status === 'connecting' ? (
                    <div className="text-center py-8 text-muted-foreground">
                      <Loader2 className="w-8 h-8 mx-auto mb-4 animate-spin text-blue-500" />
                      <p>Searching for microsegments...</p>
                      <p className="text-sm">Results will appear as they are found.</p>
                    </div>
                  ) : streamState.status === 'complete' && streamState.microsegments.length === 0 ? (
                    <div className="text-center py-12 text-muted-foreground">
                      <Layers className="w-12 h-12 mx-auto mb-4 opacity-50" />
                      <p>No microsegments found that improve over individual features.</p>
                      <p className="text-sm">Try increasing max depth or adjusting thresholds.</p>
                    </div>
                  ) : null}
                </>
              ) : (
                /* Original non-streaming mode with lazy loading */
                <>
                  {microsegmentsLoading && allMicrosegments.length === 0 ? (
                    <div className="flex items-center justify-center py-12">
                      <Loader2 className="w-8 h-8 animate-spin text-blue-500" />
                    </div>
                  ) : allMicrosegments.length > 0 ? (
                    <div className="space-y-4">
                      <div className="flex items-center justify-between">
                        <span className="text-sm text-muted-foreground">
                          Showing {allMicrosegments.length} of {totalMicrosegments} microsegments
                        </span>
                        {hasNextPage && (
                          <span className="text-xs text-muted-foreground">
                            Scroll down to load more
                          </span>
                        )}
                      </div>

                      {allMicrosegments.map((micro, i) => (
                        <Card key={`${micro.name}-${i}`} className="bg-slate-50 dark:bg-slate-900">
                          <CardContent className="pt-4">
                            <div className="flex items-start justify-between">
                              <div className="flex-1">
                                <div className="flex items-center gap-2 mb-2">
                                  <h4 className="font-semibold text-sm">
                                    Microsegment #{i + 1}
                                  </h4>
                                  {micro.depth && (
                                    <Badge variant="secondary" className="text-xs">
                                      {micro.depth}-way
                                    </Badge>
                                  )}
                                </div>
                                <div className="flex flex-wrap gap-1">
                                  {micro.conditions.map((cond, j) => (
                                    <Badge key={j} variant="outline" className="text-xs">
                                      {cond}
                                    </Badge>
                                  ))}
                                </div>
                              </div>
                              <div className="text-right space-y-1">
                                <div className="text-2xl font-bold text-blue-600">
                                  x{micro.lift.toFixed(2)}
                                </div>
                                <div className="text-sm text-muted-foreground">
                                  {(micro.support * 100).toFixed(1)}% support
                                </div>
                                <div className="text-sm text-muted-foreground">
                                  {micro.support_count.toLocaleString()} records
                                </div>
                                <div className="text-xs text-muted-foreground">
                                  RIG: {micro.rig.toFixed(3)}
                                </div>
                              </div>
                            </div>
                          </CardContent>
                        </Card>
                      ))}

                      {/* Infinite scroll trigger */}
                      <div ref={loadMoreRef} className="py-4 flex justify-center">
                        {isFetchingNextPage ? (
                          <div className="flex items-center gap-2 text-muted-foreground">
                            <Loader2 className="w-5 h-5 animate-spin" />
                            <span className="text-sm">Loading more...</span>
                          </div>
                        ) : hasNextPage ? (
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => fetchNextPage()}
                          >
                            Load More
                          </Button>
                        ) : allMicrosegments.length > 0 ? (
                          <span className="text-sm text-muted-foreground">
                            All microsegments loaded
                          </span>
                        ) : null}
                      </div>
                    </div>
                  ) : (
                    <div className="text-center py-12 text-muted-foreground">
                      <Layers className="w-12 h-12 mx-auto mb-4 opacity-50" />
                      <p>No microsegments found that improve over individual features.</p>
                      <p className="text-sm">Try increasing max depth or adjusting thresholds.</p>
                    </div>
                  )}
                </>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
