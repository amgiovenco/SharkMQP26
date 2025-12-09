// Display analysis results
const AnalysisResults = ({ completedJobs, uploadedBatches, onReset }) => {
    // Group completed jobs by batch
    const jobsByBatch = {};
    completedJobs.forEach(job => {
        if (!jobsByBatch[job.batchId]) {
            jobsByBatch[job.batchId] = [];
        }
        jobsByBatch[job.batchId].push(job);
    });

    // Calculate stats
    const totalCompleted = completedJobs.filter(j => j.status === 'completed').length;
    const totalErrors = completedJobs.filter(j => j.status === 'error').length;

    return (
        <div>
            {/* Header */}
            <div className="mb-8">
                <h2 className="text-3xl font-bold text-gray-900 mb-2">Step 4: Analysis Complete</h2>
                <p className="text-gray-600">All your samples have been processed</p>
            </div>

            {/* Summary Stats */}
            <div className="mb-8 grid grid-cols-3 gap-4">
                <div className="p-5 bg-gradient-to-br from-green-50 to-emerald-50 border border-green-200 rounded-lg">
                    <p className="text-xs font-semibold text-green-600 uppercase tracking-wide mb-2">Successful</p>
                    <p className="text-4xl font-bold text-green-700">{totalCompleted}</p>
                </div>
                {totalErrors > 0 && (
                    <div className="p-5 bg-gradient-to-br from-red-50 to-rose-50 border border-red-200 rounded-lg">
                        <p className="text-xs font-semibold text-red-600 uppercase tracking-wide mb-2">Errors</p>
                        <p className="text-4xl font-bold text-red-700">{totalErrors}</p>
                    </div>
                )}
                <div className="p-5 bg-gradient-to-br from-blue-50 to-indigo-50 border border-blue-200 rounded-lg">
                    <p className="text-xs font-semibold text-blue-600 uppercase tracking-wide mb-2">Total</p>
                    <p className="text-4xl font-bold text-blue-700">{completedJobs.length}</p>
                </div>
            </div>

            {/* Results by Batch */}
            <div className="space-y-8">
                {uploadedBatches.map((batch) => (
                    <div key={batch.batchId}>
                        <div className="mb-4">
                            <h3 className="text-xl font-bold text-gray-900">{batch.fileName}</h3>
                            <p className="text-sm text-gray-600 mt-1">{batch.numSamples} samples</p>
                        </div>

                        {/* Results Grid - Responsive */}
                        <div className="w-full grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
                            {jobsByBatch[batch.batchId]?.map((job, sampleIdx) => (
                                <div key={job.id} className="border border-gray-200 rounded-lg overflow-hidden bg-white hover:shadow-lg transition-shadow">
                                    {/* Sample Header */}
                                    <div className="px-5 py-3 bg-gradient-to-r from-gray-50 to-gray-100 border-b border-gray-200">
                                        <p className="text-sm font-semibold text-gray-700">
                                            Sample {sampleIdx + 1}/{batch.numSamples}
                                        </p>
                                    </div>

                                    {/* Content */}
                                    <div className="p-5">
                                        {job.status === 'error' ? (
                                            <div className="text-center py-6">
                                                <p className="text-3xl mb-2">⚠️</p>
                                                <p className="text-red-700 text-sm font-medium">Processing Error</p>
                                                <p className="text-red-600 text-xs mt-2">
                                                    {job.result?.error || job.error || 'Processing failed'}
                                                </p>
                                            </div>
                                        ) : job.result ? (
                                            <div className="space-y-4">
                                                {/* Top Prediction */}
                                                <div className="text-center py-4 bg-gradient-to-br from-green-50 to-emerald-50 rounded-lg border border-green-200">
                                                    <p className="text-xs font-semibold text-green-600 uppercase tracking-wide mb-2">Top Prediction</p>
                                                    <p className="text-2xl font-bold text-green-700 truncate px-2">{job.result.predictions?.[0]?.species || 'Unknown'}</p>
                                                    <p className="text-xl font-bold text-green-600 mt-2">
                                                        {(
                                                            (job.result.predictions?.[0]?.confidence ?? 0) * 100
                                                        ).toFixed(1)}%
                                                    </p>
                                                </div>

                                                {/* Top 5 Predictions */}
                                                {job.result.predictions && job.result.predictions.length > 0 && (
                                                    <div>
                                                        <p className="text-xs font-semibold text-gray-600 uppercase tracking-wide mb-3">Top 5 Predictions</p>
                                                        <div className="space-y-2">
                                                            {job.result.predictions.slice(0, 5).map((pred, idx) => (
                                                                <div key={idx} className="space-y-1">
                                                                    <div className="flex items-center justify-between">
                                                                        <p className="text-xs font-medium text-gray-800 flex-1 truncate pr-2">
                                                                            {idx + 1}. {pred.species}
                                                                        </p>
                                                                        <p className="text-xs font-bold text-gray-700 whitespace-nowrap">
                                                                            {(pred.confidence * 100).toFixed(1)}%
                                                                        </p>
                                                                    </div>
                                                                    <div className="w-full bg-gray-200 rounded-full h-2 overflow-hidden">
                                                                        <div
                                                                            className={`h-2 rounded-full transition-all ${
                                                                                idx === 0 ? 'bg-green-500' :
                                                                                idx === 1 ? 'bg-blue-500' :
                                                                                idx === 2 ? 'bg-purple-500' :
                                                                                'bg-gray-400'
                                                                            }`}
                                                                            style={{ width: `${pred.confidence * 100}%` }}
                                                                        />
                                                                    </div>
                                                                </div>
                                                            ))}
                                                        </div>
                                                    </div>
                                                )}
                                            </div>
                                        ) : (
                                            <div className="text-center py-6">
                                                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto mb-2"></div>
                                                <p className="text-gray-600 text-sm">Processing...</p>
                                            </div>
                                        )}
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                ))}
            </div>

            {/* CTA */}
            <div className="mt-10 flex gap-3">
                <button
                    onClick={onReset}
                    className="flex-1 px-6 py-3 bg-blue-600 text-white font-medium rounded-lg hover:bg-blue-700 transition"
                >
                    Start New Analysis
                </button>
            </div>
        </div>
    );
};

export default AnalysisResults;
