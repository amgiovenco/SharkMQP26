const API_BASE_URL = process.env.REACT_APP_API_BASE_URL;

export async function apiFetch(endpoint, options = {}) {
    const token = localStorage.getItem("jwt");

    const headers = {
        "Content-Type": "application/json",
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
        ...options.headers,
    };

    try {
        const response = await fetch(`${API_BASE_URL}${endpoint}`, {
            ...options,
            headers,
        });

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
        console.error("apiFetch error:", err);
        throw err;
    }
}