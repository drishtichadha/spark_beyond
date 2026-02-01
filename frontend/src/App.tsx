import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Layout } from './components/layout/Layout';
import { Dashboard } from './pages/Dashboard';
import { DataOverview } from './pages/DataOverview';
import { FeatureEngineering } from './pages/FeatureEngineering';
import { ModelTraining } from './pages/ModelTraining';
import { ModelComparison } from './pages/ModelComparison';
import { Insights } from './pages/Insights';
import { SessionProvider } from './contexts/SessionContext';
import { setupSessionInterceptor } from './lib/api';
import './index.css';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 1000 * 60 * 5, // 5 minutes - backend caches expensive computations
      retry: 1,
    },
  },
});

// Setup session interceptor for API calls
setupSessionInterceptor();

function App() {
  return (
    <SessionProvider>
      <QueryClientProvider client={queryClient}>
        <BrowserRouter>
          <Routes>
            <Route path="/" element={<Layout />}>
              <Route index element={<Dashboard />} />
              <Route path="data" element={<DataOverview />} />
              <Route path="features" element={<FeatureEngineering />} />
              <Route path="training" element={<ModelTraining />} />
              <Route path="comparison" element={<ModelComparison />} />
              <Route path="insights" element={<Insights />} />
            </Route>
          </Routes>
        </BrowserRouter>
      </QueryClientProvider>
    </SessionProvider>
  );
}

export default App;
