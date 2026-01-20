import { useState, useCallback, useRef, useEffect } from 'react';
import type { Microsegment } from './api';
import { getWebSocketUrl } from './api';

export type DiscoveryStatus = 'idle' | 'connecting' | 'discovering' | 'complete' | 'error' | 'cancelled';

export interface DiscoveryProgress {
  progress: number;
  message: string;
}

export interface MicrosegmentStreamState {
  status: DiscoveryStatus;
  progress: DiscoveryProgress;
  microsegments: Microsegment[];
  totalFound: number;
  error: string | null;
}

export interface MicrosegmentStreamParams {
  minSupport: number;
  minLift: number;
  maxDepth: number;
  topNFeatures: number;
  maxMicrosegments: number;
}

export interface UseMicrosegmentStreamReturn {
  state: MicrosegmentStreamState;
  startDiscovery: (params: MicrosegmentStreamParams) => void;
  cancelDiscovery: () => void;
  resetState: () => void;
  isConnected: boolean;
}

const initialState: MicrosegmentStreamState = {
  status: 'idle',
  progress: { progress: 0, message: '' },
  microsegments: [],
  totalFound: 0,
  error: null,
};

export function useMicrosegmentStream(): UseMicrosegmentStreamReturn {
  const [state, setState] = useState<MicrosegmentStreamState>(initialState);
  const [isConnected, setIsConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
    };
  }, []);

  const connect = useCallback((): Promise<WebSocket> => {
    return new Promise((resolve, reject) => {
      // Close existing connection if any
      if (wsRef.current) {
        wsRef.current.close();
      }

      const ws = new WebSocket(getWebSocketUrl('/ws/microsegments'));

      ws.onopen = () => {
        setIsConnected(true);
        resolve(ws);
      };

      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        setIsConnected(false);
        reject(error);
      };

      ws.onclose = () => {
        setIsConnected(false);
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          handleMessage(data);
        } catch (e) {
          console.error('Failed to parse WebSocket message:', e);
        }
      };

      wsRef.current = ws;
    });
  }, []);

  const handleMessage = useCallback((data: any) => {
    switch (data.type) {
      case 'status':
        setState((prev) => ({
          ...prev,
          status: data.status === 'starting' ? 'discovering' : prev.status,
          progress: { progress: 0, message: data.message },
        }));
        break;

      case 'progress':
        setState((prev) => ({
          ...prev,
          status: 'discovering',
          progress: {
            progress: data.progress,
            message: data.message,
          },
        }));
        break;

      case 'batch':
        setState((prev) => {
          const newMicrosegments = [...prev.microsegments, ...data.microsegments];
          return {
            ...prev,
            microsegments: newMicrosegments,
            totalFound: newMicrosegments.length,
          };
        });
        break;

      case 'complete':
        setState((prev) => ({
          ...prev,
          status: 'complete',
          progress: { progress: 100, message: data.message || 'Discovery complete!' },
          totalFound: data.total,
        }));
        // Close connection after completion
        if (wsRef.current) {
          wsRef.current.close();
        }
        break;

      case 'error':
        setState((prev) => ({
          ...prev,
          status: 'error',
          error: data.message,
        }));
        break;

      case 'cancelled':
        setState((prev) => ({
          ...prev,
          status: 'cancelled',
          progress: { ...prev.progress, message: 'Discovery cancelled' },
        }));
        break;

      case 'pong':
        // Heartbeat response, ignore
        break;

      default:
        console.log('Unknown message type:', data.type);
    }
  }, []);

  const startDiscovery = useCallback(async (params: MicrosegmentStreamParams) => {
    // Reset state before starting
    setState({
      ...initialState,
      status: 'connecting',
      progress: { progress: 0, message: 'Connecting...' },
    });

    try {
      const ws = await connect();

      setState((prev) => ({
        ...prev,
        status: 'discovering',
        progress: { progress: 0, message: 'Starting discovery...' },
      }));

      // Send start command
      ws.send(
        JSON.stringify({
          action: 'start',
          params: {
            min_support: params.minSupport,
            min_lift: params.minLift,
            max_depth: params.maxDepth,
            top_n_features: params.topNFeatures,
            max_microsegments: params.maxMicrosegments,
          },
        })
      );
    } catch (error) {
      setState((prev) => ({
        ...prev,
        status: 'error',
        error: 'Failed to connect to server',
      }));
    }
  }, [connect]);

  const cancelDiscovery = useCallback(() => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ action: 'cancel' }));
    }
  }, []);

  const resetState = useCallback(() => {
    setState(initialState);
    if (wsRef.current) {
      wsRef.current.close();
    }
  }, []);

  return {
    state,
    startDiscovery,
    cancelDiscovery,
    resetState,
    isConnected,
  };
}
