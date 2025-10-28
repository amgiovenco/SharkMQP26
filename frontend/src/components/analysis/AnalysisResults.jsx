// Display analysis results
const AnalysisResults = ({ completedJobs, uploadedFiles, onReset }) => {
    return (
        <div>
            <h2 className="text-xl font-semibold mb-4">Step 4: Results</h2>

            <div className="mb-6 p-4 border rounded bg-green-50">
                <p className="text-green-800 font-medium">
                    All {completedJobs.length} analyses complete!
                </p>
            </div>

            <div className="space-y-6 mb-6">
                {completedJobs.map((job, i) => (
                    <div key={job.id} className="p-4 border-2 border-blue-200 rounded bg-blue-50">
                        <h3 className="font-bold text-lg mb-3">Analysis {i + 1}: {uploadedFiles[i]?.name}</h3>

                        {job.status === 'error' ? (
                            <div className="p-3 border border-red-300 bg-red-50 rounded">
                                <p className="text-red-700 text-sm">
                                    <strong>Error:</strong> {job.result?.error || job.error || 'Processing failed'}
                                </p>
                            </div>
                        ) : job.result ? (
                            <div>
                                {/* Winner Section */}
                                <div className="mb-4 p-3 border-l-4 border-green-500 bg-green-50 rounded">
                                    <p className="text-sm text-gray-600">Top Prediction</p>
                                    <p className="text-2xl font-bold text-green-700">{job.result.winner}</p>
                                    <p className="text-lg font-semibold text-green-600 mt-1">
                                        {(job.result.confidence * 100).toFixed(1)}% confidence
                                    </p>
                                </div>

                                {/* Top 5 Predictions */}
                                {job.result.topk && job.result.topk.length > 0 && (
                                    <div>
                                        <p className="text-sm font-semibold text-gray-700 mb-2">Top 5 Predictions:</p>
                                        <div className="space-y-2">
                                            {job.result.topk.map((pred, idx) => (
                                                <div key={idx} className="flex items-center justify-between p-2 bg-white rounded border border-gray-200">
                                                    <div className="flex-1">
                                                        <p className="text-sm font-medium text-gray-800">{idx + 1}. {pred.label}</p>
                                                    </div>
                                                    <div className="w-24">
                                                        <div className="w-full bg-gray-200 rounded h-2">
                                                            <div
                                                                className={`h-2 rounded transition-all ${
                                                                    idx === 0 ? 'bg-green-500' :
                                                                    idx === 1 ? 'bg-blue-500' :
                                                                    'bg-gray-400'
                                                                }`}
                                                                style={{ width: `${pred.prob * 100}%` }}
                                                            />
                                                        </div>
                                                    </div>
                                                    <p className="text-sm font-semibold text-gray-700 ml-2 w-14 text-right">
                                                        {(pred.prob * 100).toFixed(1)}%
                                                    </p>
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                )}
                            </div>
                        ) : (
                            <div className="p-3 border border-yellow-300 bg-yellow-50 rounded">
                                <p className="text-yellow-700 text-sm">Processing results...</p>
                            </div>
                        )}
                    </div>
                ))}
            </div>

            <button
                onClick={onReset}
                className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
            >
                Start New Analysis
            </button>
        </div>
    );
};

export default AnalysisResults;
