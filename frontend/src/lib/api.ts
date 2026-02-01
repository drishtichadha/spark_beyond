import axios from 'axios';
import { getSessionId } from '../contexts/SessionContext';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Session ID header name
const SESSION_HEADER = 'X-Session-ID';

/**
 * Setup session interceptor to add X-Session-ID header to all requests
 */
export function setupSessionInterceptor() {
  api.interceptors.request.use((config) => {
    const sessionId = getSessionId();
    if (sessionId) {
      config.headers[SESSION_HEADER] = sessionId;
    }
    return config;
  });

  // Response interceptor to handle session-related responses
  api.interceptors.response.use(
    (response) => {
      // Check if server returned a new session ID
      const newSessionId = response.headers[SESSION_HEADER.toLowerCase()];
      const isNewSession = response.headers['x-session-new'];

      if (newSessionId && isNewSession === 'true') {
        console.log('[Session] Server created new session:', newSessionId);
        // Store the server-assigned session ID
        localStorage.setItem('spark_tune_session_id', newSessionId);
      }

      return response;
    },
    (error) => {
      // Handle session expiry or invalid session
      if (error.response?.status === 401) {
        const detail = error.response?.data?.detail;
        if (detail === 'Session expired' || detail === 'Invalid session') {
          console.warn('[Session] Session invalid, clearing...');
          localStorage.removeItem('spark_tune_session_id');
          // Reload to get a new session
          window.location.reload();
        }
      }
      return Promise.reject(error);
    }
  );
}

/**
 * Get WebSocket URL with session ID as query parameter
 */
export function getWebSocketUrl(path: string): string {
  const wsBase = import.meta.env.VITE_WS_URL || 'ws://localhost:8000';
  const sessionId = getSessionId();
  const url = new URL(path, wsBase);
  if (sessionId) {
    url.searchParams.set('session_id', sessionId);
  }
  return url.toString();
}

// Types
export interface APIResponse<T = unknown> {
  success: boolean;
  message: string;
  data?: T;
}

export interface DatasetInfo {
  rows: number;
  columns: number;
  column_names: string[];
  column_types: Record<string, string>;
}

export interface QualityReport {
  quality_score: number;
  row_count: number;
  column_count: number;
  duplicate_count: number;
  issues: Array<{ column: string; issue: string; severity: string }>;
  recommendations: Array<{ column: string; action: string; priority: string }>;
}

export interface SchemaColumn {
  name: string;
  distinct_count?: number;
  count?: number;
  null_count?: number;
  mean?: number;
  std?: number;
  min?: number;
  max?: number;
}

export interface SchemaInfo {
  categorical: SchemaColumn[];
  numerical: SchemaColumn[];
  boolean: SchemaColumn[];
}

export interface FeatureSummary {
  original_features: number;
  total_features: number;
  generated_features: number;
  feature_categories: Record<string, number>;
  sample_features: Record<string, string[]>;
}

export interface TrainingMetrics {
  accuracy?: number;
  precision?: number;
  recall?: number;
  f1_score?: number;
  auc_roc?: number;
}

export interface FeatureImportance {
  feature: string;
  importance: number;
}

export interface InsightItem {
  Condition: string;
  Lift: string;
  Lift_Value: number;
  Support: string;
  Support_Count: string;
  Support_Value: number;
  RIG: string;
  RIG_Value: number;
  Class_Rate: string;
}

export interface Microsegment {
  name: string;
  conditions: string[];
  lift: number;
  support: number;
  support_count: number;
  rig: number;
  depth?: number;
  features_involved?: string[];
  description?: string;
}

export interface PaginationInfo {
  page: number;
  page_size: number;
  total: number;
  total_pages: number;
  has_next: boolean;
  has_prev: boolean;
}

export interface SortInfo {
  sort_by: 'lift' | 'support' | 'rig' | 'support_count';
  sort_order: 'asc' | 'desc';
}

export interface MicrosegmentsPaginatedResponse {
  microsegments: Microsegment[];
  pagination: PaginationInfo;
  sort: SortInfo;
}

export interface InsightAnalysis {
  target_class: string;
  baseline_rate: number;
  total_count: number;
  insights: InsightItem[];
  microsegments: Microsegment[];
  summary: Record<string, unknown>;
}

export interface PipelineState {
  has_data: boolean;
  has_problem: boolean;
  has_features: boolean;
  has_preprocessed: boolean;
  has_model: boolean;
  dataset_info?: DatasetInfo;
  problem?: {
    target: string;
    type: string;
    desired_result?: string;
  };
}

// Data Profiler types
export interface ProfileSummary {
  n_rows: number;
  n_columns: number;
  missing_cells: number;
  missing_cells_pct: number;
  duplicate_rows?: number;
  duplicate_rows_pct?: number;
  memory_size?: number;
  types: Record<string, number>;
}

export interface ProfileAlert {
  column: string;
  type: string;
  message: string;
}

export interface ProfileRecommendation {
  column: string;
  issue: string;
  action: string;
  priority: string;
}

export interface ProfileReport {
  summary: ProfileSummary;
  missing_values: Record<string, number>;
  alerts: ProfileAlert[];
  sample_info: {
    sampled: boolean;
    original_rows: number;
    sampled_rows?: number;
    sample_ratio?: number;
  };
  recommendations: ProfileRecommendation[];
  variables_count?: number;
}

// Model Comparison types
export interface ModelExperiment {
  experiment_id: string;
  name: string;
  model: string;
  feature_set: string;
  accuracy?: number;
  precision?: number;
  recall?: number;
  f1?: number;
  training_time: number;
  num_features?: number;
}

export interface FeatureImprovements {
  feature_engineering?: {
    original_score: number;
    engineered_score: number;
    absolute_improvement: number;
    percentage_improvement: number;
    original_model: string;
    engineered_model: string;
  };
  vs_baseline?: {
    baseline_score: number;
    best_score: number;
    absolute_improvement: number;
    percentage_improvement: number;
    baseline_model: string;
    best_model: string;
  };
  model_selection_original?: {
    worst_score: number;
    best_score: number;
    absolute_improvement: number;
    percentage_improvement: number;
    worst_model: string;
    best_model: string;
  };
  model_selection_engineered?: {
    worst_score: number;
    best_score: number;
    absolute_improvement: number;
    percentage_improvement: number;
    worst_model: string;
    best_model: string;
  };
}

export interface FeatureComparisonResult {
  base_features: {
    decision_tree?: { metrics: Record<string, number>; training_time: number; num_features: number };
    xgboost?: { metrics: Record<string, number>; training_time: number; num_features: number };
  };
  engineered_features: {
    decision_tree?: { metrics: Record<string, number>; training_time: number; num_features: number };
    xgboost?: { metrics: Record<string, number>; training_time: number; num_features: number };
  };
  improvements: FeatureImprovements;
  comparison_table: ModelExperiment[];
  best_model: {
    name: string;
    metrics: Record<string, number>;
    feature_set: string;
  };
}

// API Functions
export const dataApi = {
  getState: () => api.get<APIResponse<PipelineState>>('/api/data/state'),
  loadData: (filePath: string) => api.post<APIResponse<DatasetInfo>>('/api/data/load', { file_path: filePath }),
  setProblem: (target: string, type: string, desiredResult?: string, dateColumn?: string) => {
    const body: Record<string, string> = { target, type };
    if (desiredResult !== undefined && desiredResult !== '') {
      body.desired_result = desiredResult;
    }
    if (dateColumn !== undefined && dateColumn !== '') {
      body.date_column = dateColumn;
    }
    return api.post<APIResponse<SchemaInfo>>('/api/data/problem', body);
  },
  runQualityCheck: () => api.post<APIResponse<QualityReport>>('/api/data/quality-check'),
  getColumns: () => api.get<APIResponse<DatasetInfo>>('/api/data/columns'),
  reset: () => api.post<APIResponse>('/api/data/reset'),
  runProfile: (minimal?: boolean, maxRows?: number) =>
    api.post<APIResponse<ProfileReport>>('/api/data/profile', null, {
      params: { minimal, max_rows: maxRows },
    }),
};

export const featuresApi = {
  generate: (options: {
    include_numerical?: boolean;
    include_interactions?: boolean;
    include_binning?: boolean;
    include_datetime?: boolean;
    include_string?: boolean;
  }) => api.post<APIResponse<FeatureSummary>>('/api/features/generate', options),

  preprocess: (options: {
    imputation_strategy?: string;
    handle_outliers?: boolean;
    outlier_strategy?: string;
    outlier_threshold?: number;
    apply_scaling?: boolean;
    scaling_strategy?: string;
    group_rare?: boolean;
    rare_threshold?: number;
  }) => api.post<APIResponse>('/api/features/preprocess', options),

  getSummary: () => api.get<APIResponse>('/api/features/summary'),
};

export const modelsApi = {
  train: (options: {
    train_split?: number;
    max_depth?: number;
    learning_rate?: number;
    num_rounds?: number;
  }) => api.post<APIResponse<{ train_metrics: TrainingMetrics; test_metrics: TrainingMetrics }>>('/api/models/train', options),

  getMetrics: () => api.get<APIResponse<{ train: TrainingMetrics; test: TrainingMetrics }>>('/api/models/metrics'),

  trainBaselines: () => api.post<APIResponse>('/api/models/train-baselines'),

  runAutoML: (options: { timeout?: number; cpu_limit?: number; quick_mode?: boolean }) =>
    api.post<APIResponse>('/api/models/automl', options),

  compareFeatures: () => api.post<APIResponse<FeatureComparisonResult>>('/api/models/compare-features'),

  getComparisonSummary: () => api.get<APIResponse>('/api/models/comparison-summary'),
};

export const insightsApi = {
  getFeatureImportance: (topN?: number) =>
    api.get<APIResponse<FeatureImportance[]>>('/api/insights/feature-importance', {
      params: { top_n: topN },
    }),

  getProbabilityImpact: (topN?: number) =>
    api.get<APIResponse>('/api/insights/probability-impact', {
      params: { top_n: topN },
    }),

  getLiftSupport: (
    minSupport?: number,
    minLift?: number,
    maxDepth?: number,
    topNFeatures?: number,
    maxMicrosegments?: number
  ) =>
    api.get<APIResponse<InsightAnalysis>>('/api/insights/lift-support', {
      params: {
        min_support: minSupport,
        min_lift: minLift,
        max_depth: maxDepth,
        top_n_features: topNFeatures,
        max_microsegments: maxMicrosegments,
      },
    }),

  getMicrosegmentsPaginated: (params: {
    minSupport?: number;
    minLift?: number;
    page?: number;
    pageSize?: number;
    sortBy?: 'lift' | 'support' | 'rig' | 'support_count';
    sortOrder?: 'asc' | 'desc';
  }) =>
    api.get<APIResponse<MicrosegmentsPaginatedResponse>>('/api/insights/microsegments', {
      params: {
        min_support: params.minSupport,
        min_lift: params.minLift,
        page: params.page,
        page_size: params.pageSize,
        sort_by: params.sortBy,
        sort_order: params.sortOrder,
      },
    }),

  getShapAnalysis: (sampleSize?: number) =>
    api.get<APIResponse>('/api/insights/shap', {
      params: { sample_size: sampleSize },
    }),
};
