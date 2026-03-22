import { useState } from 'react';
import { apiFetch } from '../utility/ApiFetch';
import { useJobStatusListener } from '../components/analysis/useJobStatusListener';
import AnalysisResults from '../components/analysis/AnalysisResults';
import FileUploader from '../components/analysis/FileUploader';

const AnalysisPage = () => {
    // Step state: 'upload-files' | 'processing' | 'results'
    const [step, setStep] = useState('upload-files');
    const [uploadedFiles, setUploadedFiles] = useState([]);
    const [uploadedBatches, setUploadedBatches] = useState([]);
    const [processingJobs, setProcessingJobs] = useState([]);
    const [completedJobs, setCompletedJobs] = useState([]);
    const [totalSamplesCount, setTotalSamplesCount] = useState(0);
    const [error, setError] = useState(null);
    const [isUploading, setIsUploading] = useState(false);
    const [uploadProgress, setUploadProgress] = useState(0);

    useJobStatusListener(
        processingJobs,
        setProcessingJobs,
        setCompletedJobs,
        () => setStep('results')
    );

    const handleUpload = () => {
        if (uploadedFiles.length === 0) return;

        setError(null);
        setIsUploading(true);
        setUploadProgress(0);
        setProcessingJobs([]);
        setCompletedJobs([]);
        setUploadedBatches([]);

        const newJobs = [];
        const newBatches = [];
        let uploadCount = 0;
        const failedUploads = [];

        uploadedFiles.forEach((file) => {
            const formData = new FormData();
            formData.append('file', file);

            apiFetch('/jobs/upload', {
                method: 'POST',
                body: formData,
            })
            .then(responseData => {
                const batchId = responseData.batch_id;
                const numSamples = responseData.num_samples;
                const jobIds = responseData.job_ids;

                jobIds.forEach((jobId, sampleIdx) => {
                    newJobs.push({
                        id: jobId,
                        batchId: batchId,
                        sampleIndex: sampleIdx,
                        fileName: file.name,
                        status: 'queued',
                        result: null,
                    });
                });

                newBatches.push({
                    batchId: batchId,
                    fileName: file.name,
                    numSamples: numSamples,
                    jobIds: jobIds,
                });

                uploadCount++;
                setUploadProgress(Math.round((uploadCount / uploadedFiles.length) * 100));

                if (uploadCount === uploadedFiles.length) {
                    if (failedUploads.length > 0) {
                        setError(`${failedUploads.length} file(s) failed to upload. Proceeding with ${newBatches.length} successful uploads (${newJobs.length} total samples).`);
                    }
                    setIsUploading(false);
                    setProcessingJobs(newJobs);
                    setUploadedBatches(newBatches);
                    setTotalSamplesCount(newJobs.length);
                    if (newJobs.length > 0) {
                        setStep('processing');
                    } else {
                        setError('All files failed to upload. Please try again.');
                        setIsUploading(false);
                    }
                }
            })
            .catch(err => {
                uploadCount++;
                failedUploads.push(file.name);
                setUploadProgress(Math.round((uploadCount / uploadedFiles.length) * 100));

                if (uploadCount === uploadedFiles.length) {
                    if (failedUploads.length === uploadedFiles.length) {
                        setError(`All file uploads failed: ${err.message}`);
                    } else if (failedUploads.length > 0) {
                        setError(`${failedUploads.length} file(s) failed: ${failedUploads.join(', ')}`);
                    }
                    setIsUploading(false);
                    setProcessingJobs(newJobs);
                    setUploadedBatches(newBatches);
                    if (newJobs.length > 0) {
                        setStep('processing');
                    }
                }
            });
        });
    };

    const removeFile = (index) => {
        setUploadedFiles(prev => prev.filter((_, i) => i !== index));
    };

    const handleReset = () => {
        setStep('upload-files');
        setUploadedFiles([]);
        setUploadedBatches([]);
        setProcessingJobs([]);
        setCompletedJobs([]);
        setTotalSamplesCount(0);
        setError(null);
        setIsUploading(false);
        setUploadProgress(0);
    };

    const clearError = () => {
        setError(null);
    };

    return (
        <div className="w-full max-w-6xl mx-auto p-8">
            <h1 className="text-3xl font-bold mb-8">Analysis</h1>

            {error && (
                <div className="mb-6 p-4 border border-red-300 bg-red-50 flex justify-between items-center">
                    <p className="text-red-800">{error}</p>
                    <button
                        onClick={clearError}
                        className="text-red-800 hover:text-red-900 font-semibold"
                    >
                        ✕
                    </button>
                </div>
            )}

            {/* Step 1: Upload Files */}
            {step === 'upload-files' && (
                <div>
                    {!isUploading ? (
                        <>
                            <FileUploader
                                uploadedFiles={uploadedFiles}
                                onFilesChange={setUploadedFiles}
                                onRemoveFile={removeFile}
                            />

                            <div className="flex gap-2 mt-6">
                                <button
                                    onClick={handleUpload}
                                    disabled={uploadedFiles.length === 0}
                                    className="px-6 py-3 bg-green-600 text-white font-medium rounded-lg hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed transition"
                                >
                                    Upload & Analyze
                                </button>
                            </div>
                        </>
                    ) : (
                        <div className="space-y-4">
                            <div className="p-4 border rounded bg-blue-50">
                                <p className="text-sm font-semibold text-blue-800 mb-3">
                                    Uploading {uploadedFiles.length} file(s)...
                                </p>
                                <div className="space-y-2">
                                    {uploadedFiles.map((file, idx) => (
                                        <div key={idx}>
                                            <p className="text-sm text-gray-700 mb-1">{file.name}</p>
                                            <div className="w-full bg-gray-300 rounded h-2">
                                                <div className="bg-blue-600 h-2 rounded w-full animate-pulse" />
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            </div>
                            <div className="p-4 border-2 border-blue-400 rounded bg-blue-50">
                                <p className="text-center text-blue-700 font-medium">
                                    Overall Progress: {uploadProgress}%
                                </p>
                                <div className="w-full bg-gray-300 rounded h-3 mt-2">
                                    <div
                                        className="bg-green-600 h-3 rounded transition-all"
                                        style={{ width: `${uploadProgress}%` }}
                                    />
                                </div>
                            </div>
                        </div>
                    )}
                </div>
            )}

            {/* Step 2: Processing */}
            {step === 'processing' && (
                <div>
                    <h2 className="text-2xl font-bold mb-2">Processing</h2>
                    <p className="text-gray-600 mb-6">Your samples are being analyzed</p>

                    <div className="mb-8 p-6 bg-gradient-to-r from-blue-50 to-indigo-50 border border-blue-200 rounded-lg">
                        {(() => {
                            const completed = completedJobs.length;
                            const running = processingJobs.filter(j => j.status === 'running').length;
                            const queued = processingJobs.filter(j => j.status === 'queued').length;
                            const progress = totalSamplesCount > 0 ? (completed / totalSamplesCount) * 100 : 0;

                            return (
                                <>
                                    <div className="grid grid-cols-4 gap-4">
                                        <div>
                                            <p className="text-xs font-semibold text-gray-600 uppercase tracking-wide mb-1">Total</p>
                                            <p className="text-3xl font-bold text-gray-900">{totalSamplesCount}</p>
                                        </div>
                                        <div>
                                            <p className="text-xs font-semibold text-gray-600 uppercase tracking-wide mb-1">Completed</p>
                                            <p className="text-3xl font-bold text-green-600">{completed}</p>
                                        </div>
                                        <div>
                                            <p className="text-xs font-semibold text-gray-600 uppercase tracking-wide mb-1">Processing</p>
                                            <p className="text-3xl font-bold text-blue-600">{running}</p>
                                        </div>
                                        <div>
                                            <p className="text-xs font-semibold text-gray-600 uppercase tracking-wide mb-1">Queued</p>
                                            <p className="text-3xl font-bold text-yellow-600">{queued}</p>
                                        </div>
                                    </div>

                                    <div className="mt-6">
                                        <div className="flex justify-between items-center mb-2">
                                            <p className="text-sm font-semibold text-gray-700">Overall Progress</p>
                                            <p className="text-sm font-bold text-gray-900">{Math.round(progress)}%</p>
                                        </div>
                                        <div className="w-full bg-gray-300 rounded-full h-3 overflow-hidden">
                                            <div
                                                className="h-3 rounded-full bg-gradient-to-r from-green-400 to-green-600 transition-all duration-300"
                                                style={{ width: `${progress}%` }}
                                            />
                                        </div>
                                    </div>
                                </>
                            );
                        })()}
                    </div>

                    <div className="space-y-5">
                        {uploadedBatches.map((batch) => {
                            const batchJobs = processingJobs.filter(j => j.batchId === batch.batchId);
                            const batchCompleted = batchJobs.filter(j => j.status === 'completed').length;
                            const batchProgress = (batchCompleted / batch.numSamples) * 100;

                            return (
                                <div key={batch.batchId} className="border border-gray-200 rounded-lg p-5 bg-white hover:shadow-sm transition">
                                    <div className="flex justify-between items-start mb-3">
                                        <div>
                                            <h3 className="font-semibold text-gray-900">{batch.fileName}</h3>
                                            <p className="text-sm text-gray-600 mt-1">{batchCompleted} of {batch.numSamples} samples completed</p>
                                        </div>
                                        <span className="text-sm font-bold text-gray-700 bg-gray-100 px-3 py-1 rounded">
                                            {Math.round(batchProgress)}%
                                        </span>
                                    </div>

                                    <div className="mb-4">
                                        <div className="w-full bg-gray-200 rounded-full h-2 overflow-hidden">
                                            <div
                                                className="h-2 rounded-full bg-gradient-to-r from-blue-400 to-blue-600 transition-all duration-300"
                                                style={{ width: `${batchProgress}%` }}
                                            />
                                        </div>
                                    </div>

                                    <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-2">
                                        {batchJobs.map((job, sampleIdx) => {
                                            const isCompleted = job.status === 'completed';
                                            const isRunning = job.status === 'running';
                                            const isError = job.status === 'error';

                                            return (
                                                <div
                                                    key={job.id}
                                                    className={`p-3 rounded-lg border-2 text-center text-xs font-medium transition ${
                                                        isCompleted
                                                            ? 'bg-green-50 border-green-300 text-green-700'
                                                            : isRunning
                                                            ? 'bg-blue-50 border-blue-300 text-blue-700 animate-pulse'
                                                            : isError
                                                            ? 'bg-red-50 border-red-300 text-red-700'
                                                            : 'bg-gray-50 border-gray-300 text-gray-700'
                                                    }`}
                                                >
                                                    <p>Sample {sampleIdx + 1}</p>
                                                    {isCompleted && (
                                                        <div className="mt-1">
                                                            <p className="text-xs font-semibold truncate">{job.result?.predictions?.[0]?.species || 'Unknown'}</p>
                                                            <p className="text-xs opacity-75">{(job.result?.predictions?.[0]?.confidence * 100 || 0).toFixed(0)}%</p>
                                                        </div>
                                                    )}
                                                    {isRunning && <p className="text-xs mt-1">Processing...</p>}
                                                    {isError && <p className="text-xs mt-1">Error</p>}
                                                    {!isCompleted && !isRunning && !isError && <p className="text-xs mt-1">Waiting...</p>}
                                                </div>
                                            );
                                        })}
                                    </div>
                                </div>
                            );
                        })}
                    </div>

                    <div className="mt-8 p-4 bg-blue-50 border border-blue-200 rounded-lg">
                        <p className="text-sm text-blue-800">
                            <strong>Processing in progress...</strong> You'll be automatically taken to results when all samples complete.
                        </p>
                    </div>
                </div>
            )}

            {/* Step 3: Results */}
            {step === 'results' && (
                <AnalysisResults
                    completedJobs={completedJobs}
                    uploadedBatches={uploadedBatches}
                    onReset={handleReset}
                />
            )}
        </div>
    );
};

export default AnalysisPage;
