// when you see all the cases or something idk
import { useNavigate } from 'react-router-dom';
import { useCasesStore } from '../stores/casesStore';

const AnalysisHistoryPage = () => {
    const navigate = useNavigate();
    const { cases, isLoading, error } = useCasesStore();

    if (isLoading) {
        return (
            <div className="p-8">
                <h1 className="text-3xl font-bold mb-6">Analysis History</h1>
                <div className="text-center text-gray-500">Loading cases...</div>
            </div>
        );
    }

    if (error) {
        return (
            <div className="p-8">
                <h1 className="text-3xl font-bold mb-6">Analysis History</h1>
                <div className="text-red-500">Error: {error}</div>
            </div>
        );
    }

    if (!cases || cases.length === 0) {
        return (
            <div className="p-8">
                <h1 className="text-3xl font-bold mb-6">Analysis History</h1>
                <div className="text-gray-500">No cases found. Create one to get started.</div>
            </div>
        );
    }

    return (
        <div className="p-8">
            <h1 className="text-3xl font-bold mb-6">Analysis History</h1>

            <div className="grid gap-4">
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
        </div>
    );
};

export default AnalysisHistoryPage;