import { useEffect } from 'react';
import { useAuthStore } from '../stores/authStore';
import { usePredictionStore } from '../stores/predictionStore';

export const useAuth = () => {
    const auth = useAuthStore();
    const { clearHistory } = usePredictionStore();

    // On mount, validate JWT if it exists
    useEffect(() => {
        if (auth.jwt && !auth.isAuthenticated) {
            // JWT exists in localStorage but not in memory, re-hydrate
            auth.setAuth(auth.jwt, auth.userId, auth.username);
        }
    }, []);

    const login = async (username, password) => {
        try {
            const response = await fetch('/api/auth/login', {
                method: 'POST',
                body: JSON.stringify({ username, password }),
            });

            if (!response.ok) throw new Error('Login failed');

            const { access_token, user } = await response.json();
            
            // Set auth in store (persists to localStorage)
            auth.setAuth(access_token, user.id, user.username);

            // Fetch initial cases after login
            await fetchCases(access_token);

            return { success: true };
        } catch (error) {
            return { success: false, error: error.message };
        }
    };

    const logout = () => {
        auth.clearAuth();
        clearHistory(); // Optional: clear prediction history on logout
    };

    return {
        ...auth,
        login,
        logout,
    };
};