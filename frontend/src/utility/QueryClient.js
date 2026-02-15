import { QueryClient } from "@tanstack/react-query";
import { useAuthStore } from "../stores/authStore";

export const queryClient = new QueryClient({
    defaultOptions: {
        queries: {
            retry: (failureCount, error) => {
                // Don't retry on 401/403 errors
                if (error?.message?.includes('401') || error?.message?.includes('403')) {
                    return false;
                }
                return failureCount < 3;
            },
            onError: (error) => {
                // Global error handler for queries
                if (error?.message?.includes('Session expired')) {
                    // Already handled by apiFetch, but ensure state is cleared
                    const { clearAuth } = useAuthStore.getState();
                    clearAuth();
                }
            },
        },
        mutations: {
            retry: false,
            onError: (error) => {
                // Global error handler for mutations
                if (error?.message?.includes('Session expired')) {
                    const { clearAuth } = useAuthStore.getState();
                    clearAuth();
                }
            },
        },
    },
});