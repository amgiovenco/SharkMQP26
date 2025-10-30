import { useEffect, useState } from 'react';
import { useAuthStore } from '../stores/authStore';

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

            const response = await fetch(`/api/cases?${params}`, {
                headers: {
                    'Authorization': `Bearer ${jwt}`,
                },
            });

            if (!response.ok) throw new Error('Failed to fetch cases');

            const data = await response.json();
            setCases(data.cases);
            return data; // { page, per_page, total, cases }
        } catch (err) {
            setError(err.message);
            console.error('Error fetching cases:', err);
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