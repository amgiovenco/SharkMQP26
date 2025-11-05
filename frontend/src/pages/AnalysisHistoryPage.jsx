import { useNavigate } from 'react-router-dom';
import { useState, useEffect } from 'react';
import { apiFetch } from '../utility/ApiFetch';

// Helper to format relative dates
const formatRelativeDate = (dateString) => {
    const date = new Date(dateString);
    const now = new Date();
    const seconds = Math.floor((now - date) / 1000);

    if (seconds < 60) return 'Just now';
    const minutes = Math.floor(seconds / 60);
    if (minutes < 60) return `${minutes}m ago`;
    const hours = Math.floor(minutes / 60);
    if (hours < 24) return `${hours}h ago`;
    const days = Math.floor(hours / 24);
    if (days < 7) return `${days}d ago`;
    if (days < 30) return `${Math.floor(days / 7)}w ago`;

    return date.toLocaleDateString();
};

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
            <div className="w-full p-8">
                <h1 className="text-4xl font-bold mb-2">Case History</h1>
                <p className="text-gray-600 mb-8">All your analysis cases</p>
                <div className="text-center py-12 text-gray-500">
                    <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto mb-2"></div>
                    Loading cases...
                </div>
            </div>
        );
    }

    // Error state
    if (error) {
        return (
            <div className="w-full p-8">
                <h1 className="text-4xl font-bold mb-2">Case History</h1>
                <p className="text-gray-600 mb-8">All your analysis cases</p>
                <div className="p-4 bg-red-50 border border-red-200 rounded-lg">
                    <p className="text-red-700"><strong>Error:</strong> {error}</p>
                </div>
            </div>
        );
    }

    // No cases state
    if (!isLoading && (!cases || cases.length === 0)) {
        return (
            <div className="w-full p-8">
                <h1 className="text-4xl font-bold mb-2">Case History</h1>
                <p className="text-gray-600 mb-8">All your analysis cases</p>
                <div className="text-center py-12 bg-gray-50 rounded-lg border border-gray-200">
                    <p className="text-gray-600 text-lg">No cases found yet</p>
                    <p className="text-gray-500 text-sm mt-2">Create a new case to get started</p>
                </div>
            </div>
        );
    }

    const totalPages = Math.ceil(totalCases / casesPerPage);

    return (
        <div className="w-full p-8">
            {/* Header */}
            <div className="mb-8">
                <h1 className="text-4xl font-bold mb-2">Case History</h1>
                <p className="text-gray-600">All your analysis cases</p>
            </div>

            {/* Stats */}
            <div className="mb-8 grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="p-4 bg-blue-50 border border-blue-200 rounded-lg">
                    <p className="text-xs font-semibold text-blue-600 uppercase">Total Cases</p>
                    <p className="text-3xl font-bold text-blue-900 mt-1">{totalCases}</p>
                </div>
                <div className="p-4 bg-purple-50 border border-purple-200 rounded-lg">
                    <p className="text-xs font-semibold text-purple-600 uppercase">Page</p>
                    <p className="text-3xl font-bold text-purple-900 mt-1">{currentPage}/{totalPages}</p>
                </div>
                <div className="p-4 bg-green-50 border border-green-200 rounded-lg">
                    <p className="text-xs font-semibold text-green-600 uppercase">Showing</p>
                    <p className="text-3xl font-bold text-green-900 mt-1">{cases.length}</p>
                </div>
                <div className="p-4 bg-indigo-50 border border-indigo-200 rounded-lg">
                    <p className="text-xs font-semibold text-indigo-600 uppercase">Total Samples</p>
                    <p className="text-3xl font-bold text-indigo-900 mt-1">
                        {cases.reduce((sum, c) => sum + (c.job_ids?.length || 0), 0)}
                    </p>
                </div>
            </div>

            {/* Cases Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5 mb-8">
                {cases.map((caseItem) => (
                    <div
                        key={caseItem.id}
                        onClick={() => navigate(`/case/${caseItem.id}`)}
                        className="group bg-white border border-gray-200 rounded-lg p-5 hover:shadow-lg hover:border-blue-300 cursor-pointer transition-all"
                    >
                        {/* Header */}
                        <div className="mb-3 pb-3 border-b border-gray-100">
                            <h2 className="text-lg font-bold text-gray-900 group-hover:text-blue-600 transition truncate">
                                {caseItem.title || 'Untitled Case'}
                            </h2>
                            <p className="text-xs text-gray-500 mt-1">
                                {formatRelativeDate(caseItem.created_at)}
                            </p>
                        </div>

                        {/* Description */}
                        {caseItem.description && (
                            <p className="text-sm text-gray-600 mb-3 line-clamp-2">
                                {caseItem.description}
                            </p>
                        )}

                        {/* Subject */}
                        <div className="mb-4 p-2 bg-gray-50 rounded">
                            <p className="text-xs font-semibold text-gray-600 uppercase">Subject</p>
                            <p className="text-sm text-gray-900 font-medium">
                                {caseItem.person_name || 'Unknown'}
                            </p>
                        </div>

                        {/* Samples Count */}
                        <div className="flex items-center justify-between">
                            <div>
                                <p className="text-xs font-semibold text-gray-600 uppercase">Samples</p>
                                <p className="text-2xl font-bold text-blue-600">
                                    {caseItem.job_ids?.length || 0}
                                </p>
                            </div>
                            <div className="text-right">
                                <button className="px-3 py-1 bg-blue-600 text-white text-xs font-semibold rounded hover:bg-blue-700 transition">
                                    View →
                                </button>
                            </div>
                        </div>
                    </div>
                ))}
            </div>

            {/* Pagination */}
            {totalPages > 1 && (
                <div className="flex gap-2 justify-center items-center">
                    <button
                        onClick={() => fetchCases(currentPage - 1)}
                        disabled={currentPage === 1}
                        className="px-4 py-2 border border-gray-300 rounded-lg text-sm font-medium text-gray-700 hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed transition"
                    >
                        ← Previous
                    </button>
                    <span className="text-sm font-medium text-gray-700 px-4">
                        Page {currentPage} of {totalPages}
                    </span>
                    <button
                        onClick={() => fetchCases(currentPage + 1)}
                        disabled={currentPage >= totalPages}
                        className="px-4 py-2 border border-gray-300 rounded-lg text-sm font-medium text-gray-700 hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed transition"
                    >
                        Next →
                    </button>
                </div>
            )}
        </div>
    );
};

export default AnalysisHistoryPage;