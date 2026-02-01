import { useState, useEffect, useRef } from 'react';

/**
 * Hook that debounces a value - only updates after delay when value stops changing.
 * Useful for expensive operations triggered by user input like sliders.
 *
 * @param value The value to debounce
 * @param delay Delay in milliseconds (default: 500ms)
 * @returns The debounced value
 */
export function useDebouncedValue<T>(value: T, delay: number = 500): T {
  const [debouncedValue, setDebouncedValue] = useState<T>(value);

  useEffect(() => {
    const timer = setTimeout(() => {
      setDebouncedValue(value);
    }, delay);

    return () => {
      clearTimeout(timer);
    };
  }, [value, delay]);

  return debouncedValue;
}

/**
 * Hook that returns a debounced callback function.
 * The callback will only execute after the specified delay has passed
 * since the last invocation.
 *
 * @param callback The function to debounce
 * @param delay Delay in milliseconds (default: 500ms)
 * @returns The debounced callback
 */
export function useDebouncedCallback<T extends (...args: unknown[]) => unknown>(
  callback: T,
  delay: number = 500
): (...args: Parameters<T>) => void {
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
    };
  }, []);

  return (...args: Parameters<T>) => {
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
    }

    timeoutRef.current = setTimeout(() => {
      callback(...args);
    }, delay);
  };
}
