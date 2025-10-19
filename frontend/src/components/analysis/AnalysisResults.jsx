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

            <div className="space-y-3 mb-6">
                {completedJobs.map((job, i) => (
                    <div key={job.id} className="p-4 border rounded">
                        <p className="font-medium">Analysis {i + 1}</p>
                        <p className="text-sm text-gray-600">Status: {job.status}</p>
                        <p className="text-sm text-gray-600">File: {uploadedFiles[i]?.name}</p>
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
