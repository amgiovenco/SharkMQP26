import { useAuthStore } from '../stores/authStore';

const API_BASE_URL = process.env.REACT_APP_API_BASE_URL;

export async function apiFetch(endpoint, options = {}) {
    // Get the auth store (need to call it as a function to get current state)
    const { jwt, clearAuth } = useAuthStore.getState();
    const { skipAuthRedirect, ...fetchOptions } = options;

    const headers = {
        ...(jwt ? { Authorization: `Bearer ${jwt}` } : {}),
        ...fetchOptions.headers,
    };

    // Only set Content-Type if not FormData
    if (!(fetchOptions.body instanceof FormData)) {
        headers['Content-Type'] = 'application/json';
    }

    try {
        const response = await fetch(`${API_BASE_URL}${endpoint}`, {
            ...fetchOptions,
            headers,
        });

        // Handle unauthorized/forbidden responses - clear auth and redirect to login
        // Skip redirect for endpoints that intentionally handle 401 themselves (e.g. login)
        if ((response.status === 401 || response.status === 403) && !skipAuthRedirect) {
            clearAuth();
            window.location.href = '/';
            throw new Error('Session expired. Please log in again.');
        }

        // Handle non-2xx responses
        if (!response.ok) {
            let errorMessage = `HTTP ${response.status}`;
            try {
                const data = await response.json();
                if (data?.detail) errorMessage = data.detail;
                else if (data?.message) errorMessage = data.message;
            } catch {
                // ignore JSON parse errors
            }
            throw new Error(errorMessage);
        }

        // Try to parse JSON, otherwise return text/blob
        const contentType = response.headers.get("content-type");
        if (contentType && contentType.includes("application/json")) {
            return response.json();
        }
        return response.text();

    } catch (err) {
        throw err;
    }
}