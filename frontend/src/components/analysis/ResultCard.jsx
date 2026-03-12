import { useState, useEffect } from 'react';
import toast from 'react-hot-toast';
import MeltingCurveChart from './MeltingCurveChart';
import { apiFetch } from '../../utility/ApiFetch';

const ResultCard = ({ result, batch, onRerun }) => {
    const [expandedTopk, setExpandedTopk] = useState(false);
    const [isRerunning, setIsRerunning] = useState(false);
    const [fullResult, setFullResult] = useState(result.result);

    useEffect(() => {
        if (result.status === 'done' && result.result && !result.result.curve_data) {
            apiFetch(`/jobs/${result.id}`)
                .then(job => setFullResult(job.result_json))
                .catch(() => {});
        }
    }, [result.id, result.status, result.result]);

    if (result.status === 'queued' || result.status === 'running') {
        return (
            <div className="border rounded p-4 bg-white h-full flex flex-col justify-center items-center">
                <div className="text-center">
                    <p className="text-sm font-semibold text-gray-700 mb-2">
                        Sample {result.sampleIndex + 1}/{batch.numSamples}
                    </p>
                    <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto mb-2"></div>
                    <p className="text-sm text-gray-600">Processing...</p>
                </div>
            </div>
        );
    }

    const handleRerun = async () => {
        if (isRerunning) return;

        setIsRerunning(true);
        try {
            const response = await apiFetch(`/jobs/${result.id}/rerun`, {
                method: 'POST',
            });

            toast.success('Analysis queued for rerun');

            if (onRerun) {
                onRerun(response);
            }
        } catch (error) {
            console.error('Failed to rerun analysis:', error);
            toast.error(`Failed to rerun analysis: ${error.message}`);
        } finally {
            setIsRerunning(false);
        }
    };

    if (result.status === 'error') {
        return (
            <div className="border border-red-300 rounded p-4 bg-red-50 h-full">
                <div className="flex items-start justify-between mb-2">
                    <p className="text-sm font-semibold text-gray-700">
                        Sample {result.sampleIndex + 1}/{batch.numSamples}
                    </p>
                    <button
                        onClick={handleRerun}
                        disabled={isRerunning}
                        className="text-xs px-2 py-0.5 bg-orange-50 text-orange-600 hover:bg-orange-100 rounded border border-orange-200 transition disabled:opacity-50 disabled:cursor-not-allowed"
                        title="Rerun this analysis"
                    >
                        {isRerunning ? '...' : '↻ Retry'}
                    </button>
                </div>
                <div className="p-3 border border-red-300 bg-red-50 rounded">
                    <p className="text-red-700 text-xs">
                        <strong>Error:</strong> {fullResult?.error || 'Processing failed'}
                    </p>
                </div>
            </div>
        );
    }

    if (!fullResult) {
        return null;
    }

    // Extract top prediction from new backend format
    const topPrediction = fullResult.predictions?.[0];
    const winner = topPrediction?.species || 'Unknown';
    const confidence = topPrediction?.confidence || 0;
    const topk = fullResult.predictions?.map(p => ({ label: p.species, prob: p.confidence })) || [];
    const curve_data = fullResult.curve_data;

    const confidencePercent = (confidence * 100).toFixed(1);
    const confidenceColor = confidence > 0.9 ? 'bg-green-100 text-green-800' : confidence > 0.8 ? 'bg-yellow-100 text-yellow-800' : 'bg-red-100 text-red-800';

    return (
        <div className="border rounded p-4 bg-white shadow-sm hover:shadow-md transition h-full flex flex-col">
            {/* Header */}
            <div className="mb-3 pb-3 border-b">
                <div className="flex items-start justify-between mb-1">
                    <p className="text-xs font-semibold text-gray-600">
                        Sample {result.sampleIndex + 1}/{batch.numSamples}
                    </p>
                    <button
                        onClick={handleRerun}
                        disabled={isRerunning}
                        className="text-xs px-2 py-0.5 bg-blue-50 text-blue-600 hover:bg-blue-100 rounded border border-blue-200 transition disabled:opacity-50 disabled:cursor-not-allowed"
                        title="Rerun this analysis"
                    >
                        {isRerunning ? '...' : '↻ Rerun'}
                    </button>
                </div>
                <div className="flex items-start justify-between gap-2">
                    <h3 className="font-bold text-sm flex-1 text-gray-900">{winner}</h3>
                    <span className={`text-xs font-bold px-2 py-1 rounded whitespace-nowrap ${confidenceColor}`}>
                        {confidencePercent}%
                    </span>
                </div>
            </div>

            {/* Chart - fixed height */}
            {curve_data && (
                <div className="mb-4 w-full" style={{ height: '200px' }}>
                    <MeltingCurveChart
                        frequencies={curve_data.frequencies}
                        signal={curve_data.signal}
                        predictedSpecies={winner}
                    />
                </div>
            )}

            {/* Top 5 predictions */}
            <div className="text-xs flex-1 overflow-y-auto">
                <button
                    onClick={() => setExpandedTopk(!expandedTopk)}
                    className="font-semibold text-blue-600 hover:text-blue-800 mb-2 w-full text-left"
                >
                    {expandedTopk ? '▼ Top 5' : '▶ Top 5'}
                </button>

                {expandedTopk && topk && (
                    <div className="space-y-1 bg-gray-50 rounded p-2">
                        {topk.slice(0, 5).map((pred, idx) => (
                            <div key={idx} className="flex justify-between items-center">
                                <span className="text-gray-700 flex-1">{idx + 1}. {pred.label}</span>
                                <span className="text-gray-600 font-semibold">{(pred.prob * 100).toFixed(1)}%</span>
                            </div>
                        ))}
                    </div>
                )}
            </div>
        </div>
    );
};

export default ResultCard;
