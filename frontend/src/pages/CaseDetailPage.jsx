import { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { apiFetch } from '../utility/ApiFetch';
import { useSocket } from '../contexts/SocketContext';
import ResultCard from '../components/analysis/ResultCard';

const CaseDetailPage = () => {
    const { caseId } = useParams();
    const navigate = useNavigate();
    const socket = useSocket();

    const [caseData, setCaseData] = useState(null);
    const [results, setResults] = useState([]); // All results: {id, batchId, sampleIndex, fileName, status, result, batchInfo}
    const [uploadedBatches, setUploadedBatches] = useState([]); // Track batches
    const [isUploading, setIsUploading] = useState(false);
    const [uploadProgress, setUploadProgress] = useState(0);
    const [error, setError] = useState(null);
    const [loading, setLoading] = useState(true);

    // Load case data
    useEffect(() => {
        apiFetch(`/cases/${caseId}`)
            .then(setCaseData)
            .catch(err => {
                setError(`Failed to load case: ${err.message}`);
                setLoading(false);
            });
    }, [caseId]);

    // Load existing jobs for this case
    useEffect(() => {
        if (!caseData) return;

        apiFetch(`/cases/${caseId}/jobs?page=1&per_page=100`)
            .then(data => {
                const jobs = data.jobs || [];

                // Reconstruct batch info and results
                const batchMap = {};
                const resultsList = [];

                jobs.forEach(job => {
                    const batchId = job.batch_id || job.id;

                    if (!batchMap[batchId]) {
                        batchMap[batchId] = {
                            batchId,
                            fileName: job.file_path?.split('/').pop() || 'Unknown',
                            jobIds: [],
                        };
                    }

                    batchMap[batchId].jobIds.push(job.id);

                    resultsList.push({
                        id: job.id,
                        batchId: batchId,
                        sampleIndex: job.sample_index || 0,
                        fileName: batchMap[batchId].fileName,
                        status: job.status,
                        result: job.result_json,
                        batchInfo: batchMap[batchId],
                    });
                });

                setResults(resultsList);
                setUploadedBatches(Object.values(batchMap));
                setLoading(false);
            })
            .catch(err => {
                setError(`Failed to load jobs: ${err.message}`);
                setLoading(false);
            });
    }, [caseData, caseId]);

    // Listen for real-time job updates via socket
    useEffect(() => {
        if (!socket) return;

        const handleJobStatus = (data) => {
            const { job_id, status, result } = data;

            // Update or add result
            setResults(prevResults => {
                const existingIndex = prevResults.findIndex(r => r.id === job_id);

                if (existingIndex >= 0) {
                    // Update existing
                    const updated = [...prevResults];
                    updated[existingIndex] = {
                        ...updated[existingIndex],
                        status,
                        result,
                    };
                    return updated;
                } else {
                    // Add new (shouldn't happen but just in case)
                    return [...prevResults, {
                        id: job_id,
                        status,
                        result,
                    }];
                }
            });
        };

        socket.on('job_status', handleJobStatus);

        return () => {
            socket.off('job_status', handleJobStatus);
        };
    }, [socket]);

    // Handle file upload
    const handleFileUpload = (e) => {
        const files = Array.from(e.target.files || []);
        if (files.length === 0) return;

        setError(null);
        setIsUploading(true);
        setUploadProgress(0);

        const newBatches = [];
        const newResults = [];
        let uploadCount = 0;
        const failedUploads = [];

        files.forEach((file) => {
            const formData = new FormData();
            formData.append('file', file);
            formData.append('case_id', caseId);

            apiFetch('/jobs/upload', {
                method: 'POST',
                body: formData,
            })
                .then(responseData => {
                    const batchId = responseData.batch_id;
                    const numSamples = responseData.num_samples;
                    const jobIds = responseData.job_ids;

                    const batchInfo = {
                        batchId,
                        fileName: file.name,
                        numSamples,
                        jobIds,
                    };

                    newBatches.push(batchInfo);

                    // Create result objects for each sample
                    jobIds.forEach((jobId, sampleIdx) => {
                        newResults.push({
                            id: jobId,
                            batchId,
                            sampleIndex: sampleIdx,
                            fileName: file.name,
                            status: 'queued',
                            result: null,
                            batchInfo,
                        });
                    });

                    uploadCount++;
                    setUploadProgress(Math.round((uploadCount / files.length) * 100));

                    if (uploadCount === files.length) {
                        if (failedUploads.length > 0) {
                            setError(`${failedUploads.length} file(s) failed to upload.`);
                        }
                        setIsUploading(false);
                        setUploadedBatches(prev => [...prev, ...newBatches]);
                        setResults(prev => [...prev, ...newResults]);
                    }
                })
                .catch(err => {
                    uploadCount++;
                    failedUploads.push(file.name);
                    setUploadProgress(Math.round((uploadCount / files.length) * 100));

                    if (uploadCount === files.length) {
                        if (failedUploads.length === files.length) {
                            setError(`All uploads failed: ${err.message}`);
                        }
                        setIsUploading(false);
                    }
                });
        });
    };

    // Export results
    const handleExportAll = () => {
        const csv = generateCSV(results);
        downloadCSV(csv, `${caseData.title}_all_results.csv`);
    };

    const handleExportBatch = (batchId) => {
        const batchResults = results.filter(r => r.batchId === batchId);
        const csv = generateCSV(batchResults);
        const fileName = uploadedBatches.find(b => b.batchId === batchId)?.fileName || 'results';
        downloadCSV(csv, `${fileName}_results.csv`);
    };

    if (loading) return <div className="p-8">Loading case...</div>;
    if (!caseData) return <div className="p-8">Case not found</div>;

    const completedCount = results.filter(r => r.status === 'completed').length;
    const processingCount = results.filter(r => r.status === 'running' || r.status === 'queued').length;

    return (
        <div className="p-8 max-w-7xl mx-auto">
            {/* Header */}
            <div className="mb-8">
                <button onClick={() => navigate('/cases')} className="text-blue-600 hover:underline mb-4">
                    ← Back to Cases
                </button>
                <h1 className="text-4xl font-bold mb-2">{caseData.title}</h1>
                <p className="text-gray-600">Subject: {caseData.person_name}</p>
                <p className="text-sm text-gray-500">Created: {new Date(caseData.created_at).toLocaleDateString()}</p>
            </div>

            {/* Error message */}
            {error && (
                <div className="mb-6 p-4 border border-red-300 bg-red-50 rounded flex justify-between">
                    <p className="text-red-800">{error}</p>
                    <button onClick={() => setError(null)} className="text-red-800 hover:text-red-900 font-bold">✕</button>
                </div>
            )}

            {/* Upload Zone */}
            <div className="mb-8 p-6 border-2 border-dashed border-blue-300 rounded bg-blue-50 text-center">
                <label className="cursor-pointer">
                    <input
                        type="file"
                        multiple
                        accept=".csv"
                        onChange={handleFileUpload}
                        disabled={isUploading}
                        className="hidden"
                    />
                    <p className="text-lg font-semibold text-blue-900 mb-2">
                        {isUploading ? `Uploading (${uploadProgress}%)...` : 'Drag CSV files here or click to browse'}
                    </p>
                    <p className="text-sm text-blue-700">You can upload multiple files at once</p>
                </label>
            </div>

            {/* Stats */}
            {results.length > 0 && (
                <div className="mb-8 p-4 border rounded bg-gray-50">
                    <p className="font-semibold">
                        {completedCount} completed • {processingCount} processing • {uploadedBatches.length} file(s)
                    </p>
                </div>
            )}

            {/* Export buttons */}
            {results.length > 0 && (
                <div className="mb-8 flex gap-2">
                    <button
                        onClick={handleExportAll}
                        className="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700"
                    >
                        Export All Results
                    </button>
                </div>
            )}

            {/* Results Grid */}
            <div className="space-y-6">
                {uploadedBatches.map(batch => (
                    <div key={batch.batchId} className="border rounded p-4 bg-white">
                        <div className="flex justify-between items-start mb-4">
                            <div>
                                <h2 className="text-xl font-semibold">{batch.fileName}</h2>
                                <p className="text-sm text-gray-600">{batch.numSamples} samples</p>
                            </div>
                            <button
                                onClick={() => handleExportBatch(batch.batchId)}
                                className="px-3 py-1 bg-blue-600 text-white text-sm rounded hover:bg-blue-700"
                            >
                                Export
                            </button>
                        </div>

                        {/* Results grid - 3 columns */}
                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                            {results
                                .filter(r => r.batchId === batch.batchId)
                                .sort((a, b) => a.sampleIndex - b.sampleIndex)
                                .map(result => (
                                    <ResultCard key={result.id} result={result} batch={batch} />
                                ))}
                        </div>
                    </div>
                ))}
            </div>

            {/* Empty state */}
            {results.length === 0 && !isUploading && (
                <div className="text-center text-gray-500 py-12">
                    <p>No results yet. Upload CSV files to get started.</p>
                </div>
            )}
        </div>
    );
};

// Helper functions
const generateCSV = (results) => {
    const rows = [
        ['File', 'Sample', 'Predicted Species', 'Confidence', 'Top 2', 'Top 3', 'Top 4', 'Top 5'].join(','),
    ];

    results.forEach(r => {
        if (r.result) {
            const topPrediction = r.result.predictions?.[0];
            const topLabels = r.result.predictions?.slice(1, 5).map(p => p.species).join(' | ') || '';
            rows.push([
                r.fileName,
                r.sampleIndex + 1,
                topPrediction?.species || 'Unknown',
                ((topPrediction?.confidence || 0) * 100).toFixed(2) + '%',
                topLabels,
            ].join(','));
        }
    });

    return rows.join('\n');
};

const downloadCSV = (csv, filename) => {
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    window.URL.revokeObjectURL(url);
};

export default CaseDetailPage;
