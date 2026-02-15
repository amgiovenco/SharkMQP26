import { useEffect, useState } from 'react';
import { useAuthStore } from '../stores/authStore';
import { apiFetch } from '../utility/ApiFetch';

export const useCases = () => {
    const { jwt } = useAuthStore();
    const [cases, setCases] = useState([]);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState(null);

    const fetchCases = async (page = 1, query = null) => {
        if (!jwt) return;

        setIsLoading(true);
        setError(null);

        try {
            const params = new URLSearchParams();
            params.append('page', page);
            params.append('per_page', 20);
            if (query) params.append('q', query);

            const data = await apiFetch(`/api/cases?${params}`);
            setCases(data.cases);
            return data; // { page, per_page, total, cases }
        } catch (err) {
            setError(err.message);
        } finally {
            setIsLoading(false);
        }
    };

    // Fetch cases on mount if authenticated
    useEffect(() => { if (jwt) fetchCases() }, [jwt]);

    return {
        cases,
        isLoading,
        error,
        fetchCases,
    };
};