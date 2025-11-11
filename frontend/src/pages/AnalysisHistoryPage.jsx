// when you see all the cases or something idk
import { useNavigate } from 'react-router-dom';
import { useState, useEffect } from 'react';
import { apiFetch } from '../utility/ApiFetch';

const AnalysisHistoryPage = () => {
    const navigate = useNavigate();
    const [cases, setCases] = useState([]);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState(null);
    const [currentPage, setCurrentPage] = useState(1);
    const [totalCases, setTotalCases] = useState(0);
    const casesPerPage = 20;

    const fetchCases = async (page = 1) => {
        setIsLoading(true);
        setError(null);

        try {
            const data = await apiFetch(`/cases?page=${page}&per_page=${casesPerPage}`);
            setCases(data.cases || []);
            setTotalCases(data.total || 0);
            setCurrentPage(page);
        } catch (err) {
            setError(err.message);
        } finally {
            setIsLoading(false);
        }
    };

    // Fetch cases on mount
    useEffect(() => {
        fetchCases(1);
    }, []);

    // Loading state
    if (isLoading) {
        return (
            <div className="p-8">
                <h1 className="text-3xl font-bold mb-6">Analysis History</h1>
                <div className="text-center text-gray-500">Loading cases...</div>
            </div>
        );
    }

    // Error state
    if (error) {
        return (
            <div className="p-8">
                <h1 className="text-3xl font-bold mb-6">Analysis History</h1>
                <div className="text-red-500">Error: {error}</div>
            </div>
        );
    }

    // No cases state
    if (!isLoading && (!cases || cases.length === 0)) {
        return (
            <div className="p-8">
                <h1 className="text-3xl font-bold mb-6">Analysis History</h1>
                <div className="text-gray-500">No cases found. Create one to get started.</div>
            </div>
        );
    }

    const totalPages = Math.ceil(totalCases / casesPerPage);

    return (
        <div className="p-8">
            <h1 className="text-3xl font-bold mb-6">Analysis History</h1>

            <div className="mb-4 text-sm text-gray-600">
                Showing {cases.length > 0 ? (currentPage - 1) * casesPerPage + 1 : 0} - {Math.min(currentPage * casesPerPage, totalCases)} of {totalCases} cases
            </div>

            <div className="grid gap-4 mb-6">
                {cases.map((caseItem) => (
                    <div
                        key={caseItem.id}
                        onClick={() => navigate(`/case/${caseItem.id}`)}
                        className="p-4 border rounded-lg hover:bg-gray-50 cursor-pointer transition"
                    >
                        <div className="flex justify-between items-start">
                            <div className="flex-1">
                                <h2 className="text-lg font-semibold text-gray-900">
                                    {caseItem.title || 'Untitled Case'}
                                </h2>
                                <p className="text-sm text-gray-600 mt-1">
                                    {caseItem.description || 'No description'}
                                </p>
                                <p className="text-sm text-gray-500 mt-2">
                                    Subject: {caseItem.person_name || 'Unknown'}
                                </p>
                            </div>
                            <div className="text-right ml-4">
                                <p className="text-xs text-gray-500">
                                    {new Date(caseItem.created_at).toLocaleDateString()}
                                </p>
                                <p className="text-xs text-gray-400 mt-1">
                                    {caseItem.job_ids?.length || 0} jobs
                                </p>
                            </div>
                        </div>
                    </div>
                ))}
            </div>

            {totalPages > 1 && (
                <div className="flex gap-2 justify-center items-center">
                    <button
                        onClick={() => fetchCases(currentPage - 1)}
                        disabled={currentPage === 1}
                        className="px-3 py-1 border rounded disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-100"
                    >
                        ← Previous
                    </button>
                    <span className="text-sm text-gray-600">
                        Page {currentPage} of {totalPages}
                    </span>
                    <button
                        onClick={() => fetchCases(currentPage + 1)}
                        disabled={currentPage >= totalPages}
                        className="px-3 py-1 border rounded disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-100"
                    >
                        Next →
                    </button>
                </div>
            )}
        </div>
    );
};

export default AnalysisHistoryPage;