import { useEffect, useRef } from 'react';
import { useAuthStore } from '../stores/authStore';
import { useCasesStore } from '../stores/casesStore';
import { apiFetch } from '../utility/ApiFetch';

/**
 * Hook to initialize app on mount
 * Fetches fresh user data and cases when the app loads
 * Validates JWT and clears auth if invalid
 */
export const useInitializeApp = () => {
    const { jwt, setAuth, clearAuth } = useAuthStore();
    const { setCases, setError } = useCasesStore();
    const hasInitialized = useRef(false);

    useEffect(() => {
        // Get current auth state
        const authState = useAuthStore.getState();
        const { jwt: currentJwt } = authState;

        // Only initialize if user has a valid JWT
        if (!currentJwt) return;

        // Only run once on mount
        if (hasInitialized.current) return;
        hasInitialized.current = true;

        const initializeApp = async () => {
            try {
                // Fetch fresh user data to validate JWT
                const userData = await apiFetch('/auth/me');

                // Update auth store with fresh data
                setAuth(
                    jwt,
                    userData.id,
                    userData.username,
                    userData.role,
                    userData.first_name,
                    userData.last_name,
                    userData.job_title
                );

                // Fetch fresh cases data
                const casesData = await apiFetch('/cases');
                setCases(casesData.cases);
                setError(null);

                console.log('App initialized successfully with fresh data');
            } catch (error) {
                console.error('Failed to initialize app:', error);

                // If JWT is invalid (401 Unauthorized), clear auth
                if (error.message && (error.message.includes('401') || error.message.includes('Invalid token'))) {
                    console.log('JWT token invalid or expired, clearing auth');
                    clearAuth();
                } else {
                    // For other errors (network, server error, etc.), just log but don't clear auth
                    // User will see stale data from localStorage but can still use the app
                    setError(error.message || 'Failed to refresh data');
                }
            }
        };

        initializeApp();
    // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);
};
