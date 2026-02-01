/**
 * Session Context - Manages session ID for API persistence
 *
 * Provides:
 * - Session ID generation and storage in localStorage
 * - Session ID context for components
 * - Hook for accessing session ID
 */

import React, { createContext, useContext, useEffect, useState, ReactNode } from 'react';
import { v4 as uuidv4 } from 'uuid';

const SESSION_STORAGE_KEY = 'spark_tune_session_id';

interface SessionContextType {
  sessionId: string;
  isNewSession: boolean;
  clearSession: () => void;
}

const SessionContext = createContext<SessionContextType | null>(null);

/**
 * Validate that a string is a valid UUID v4
 */
function isValidUUID(str: string): boolean {
  const uuidRegex = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
  return uuidRegex.test(str);
}

/**
 * Get or create session ID from localStorage
 */
function getOrCreateSessionId(): { sessionId: string; isNew: boolean } {
  const existingId = localStorage.getItem(SESSION_STORAGE_KEY);

  if (existingId && isValidUUID(existingId)) {
    return { sessionId: existingId, isNew: false };
  }

  // Generate new session ID
  const newId = uuidv4();
  localStorage.setItem(SESSION_STORAGE_KEY, newId);
  return { sessionId: newId, isNew: true };
}

interface SessionProviderProps {
  children: ReactNode;
}

/**
 * Session Provider Component
 *
 * Wraps the application and provides session context to all children.
 * Must be placed near the top of the component tree.
 */
export function SessionProvider({ children }: SessionProviderProps) {
  const [sessionId, setSessionId] = useState<string>('');
  const [isNewSession, setIsNewSession] = useState<boolean>(false);
  const [isInitialized, setIsInitialized] = useState<boolean>(false);

  useEffect(() => {
    const { sessionId: id, isNew } = getOrCreateSessionId();
    setSessionId(id);
    setIsNewSession(isNew);
    setIsInitialized(true);

    if (isNew) {
      console.log('[Session] Created new session:', id);
    } else {
      console.log('[Session] Restored existing session:', id);
    }
  }, []);

  /**
   * Clear the current session and generate a new one
   */
  const clearSession = () => {
    localStorage.removeItem(SESSION_STORAGE_KEY);
    const { sessionId: newId } = getOrCreateSessionId();
    setSessionId(newId);
    setIsNewSession(true);
    console.log('[Session] Cleared session, created new:', newId);
  };

  // Don't render children until session is initialized
  if (!isInitialized) {
    return null;
  }

  return (
    <SessionContext.Provider value={{ sessionId, isNewSession, clearSession }}>
      {children}
    </SessionContext.Provider>
  );
}

/**
 * Hook to access session context
 *
 * @returns Session context with sessionId, isNewSession flag, and clearSession function
 * @throws Error if used outside of SessionProvider
 */
export function useSession(): SessionContextType {
  const context = useContext(SessionContext);
  if (!context) {
    throw new Error('useSession must be used within a SessionProvider');
  }
  return context;
}

/**
 * Get the current session ID directly from localStorage
 *
 * Useful for scenarios where hooks can't be used (e.g., API interceptors)
 */
export function getSessionId(): string | null {
  const id = localStorage.getItem(SESSION_STORAGE_KEY);
  return id && isValidUUID(id) ? id : null;
}

export default SessionContext;
