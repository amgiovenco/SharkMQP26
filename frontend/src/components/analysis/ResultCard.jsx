import { useState } from 'react';
import MeltingCurveChart from './MeltingCurveChart';

const ResultCard = ({ result, batch }) => {
    const [expandedTopk, setExpandedTopk] = useState(false);

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

    if (result.status === 'error') {
        return (
            <div className="border border-red-300 rounded p-4 bg-red-50 h-full">
                <p className="text-sm font-semibold text-gray-700 mb-2">
                    Sample {result.sampleIndex + 1}/{batch.numSamples}
                </p>
                <div className="p-3 border border-red-300 bg-red-50 rounded">
                    <p className="text-red-700 text-xs">
                        <strong>Error:</strong> {result.result?.error || 'Processing failed'}
                    </p>
                </div>
            </div>
        );
    }

    if (!result.result) {
        return null;
    }

    // Extract top prediction from new backend format
    const topPrediction = result.result.predictions?.[0];
    const winner = topPrediction?.species || 'Unknown';
    const confidence = topPrediction?.confidence || 0;
    const topk = result.result.predictions?.map(p => ({ label: p.species, prob: p.confidence })) || [];
    const curve_data = result.result.curve_data;

    const confidencePercent = (confidence * 100).toFixed(1);
    const confidenceColor = confidence > 0.8 ? 'bg-green-100 text-green-800' : confidence > 0.6 ? 'bg-yellow-100 text-yellow-800' : 'bg-red-100 text-red-800';

    return (
        <div className="border rounded p-4 bg-white shadow-sm hover:shadow-md transition h-full flex flex-col">
            {/* Header */}
            <div className="mb-3 pb-3 border-b">
                <p className="text-xs font-semibold text-gray-600 mb-1">
                    Sample {result.sampleIndex + 1}/{batch.numSamples}
                </p>
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
