// when you upload csv and then wait for resutls and then see results
import { useState } from 'react';
import { apiFetch } from '../utility/ApiFetch';
import { useCasesStore } from '../stores/casesStore';
import { usePredictionStore } from '../stores/predictionStore';
import { useJobStatusListener } from '../components/analysis/useJobStatusListener';
import AnalysisResults from '../components/analysis/AnalysisResults';
import FileUploader from '../components/analysis/FileUploader';

const AnalysisPage = () => {
    const { cases, addCase } = useCasesStore();
    const { addPrediction } = usePredictionStore();

    // Step state: 'select-case' | 'upload-files' | 'processing' | 'results'
    const [step, setStep] = useState('select-case');
    const [selectedCase, setSelectedCase] = useState(null);
    const [isCreatingCase, setIsCreatingCase] = useState(false);
    const [newCaseForm, setNewCaseForm] = useState({
        title: '',
        description: '',
        person_name: '',
    });
    const [uploadedFiles, setUploadedFiles] = useState([]);
    const [uploadedBatches, setUploadedBatches] = useState([]); // Track batches: [{batchId, fileName, numSamples, jobIds}]
    const [processingJobs, setProcessingJobs] = useState([]);
    const [completedJobs, setCompletedJobs] = useState([]);
    const [totalSamplesCount, setTotalSamplesCount] = useState(0); // Fixed total for progress tracking
    const [error, setError] = useState(null);
    const [isUploading, setIsUploading] = useState(false);
    const [uploadProgress, setUploadProgress] = useState(0);

    // Custom hook to listen for job status updates
    useJobStatusListener(
        processingJobs,
        setProcessingJobs,
        setCompletedJobs,
        () => setStep('results')
    );

    // Create new case
    const handleCreateCase = (e) => {
        e.preventDefault();
        if (!validateCaseForm()) {
            return;
        }
        apiFetch('/cases', {
            method: 'POST',
            body: JSON.stringify(newCaseForm),
        })
        .then(caseData => {
            addCase(caseData);
            setSelectedCase(caseData.id);
            setNewCaseForm({ title: '', description: '', person_name: '' });
            setIsCreatingCase(false);
            setError(null);
        })
        .catch(err => {
            setError(`Failed to create case: ${err.message}`);
        });
    };

    // Proceed to next step in the analysis workflow
    const handleNextStep = () => {
        if (step === 'select-case' && selectedCase) {
            setStep('upload-files');
            setError(null);
        } else if (step === 'upload-files' && uploadedFiles.length > 0) {
            // Upload all files
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
                formData.append('case_id', selectedCase);

                apiFetch('/jobs/upload', {
                    method: 'POST',
                    body: formData,
                })
                .then(responseData => {
                    // New response format: {status, batch_id, num_samples, job_ids}
                    const batchId = responseData.batch_id;
                    const numSamples = responseData.num_samples;
                    const jobIds = responseData.job_ids;

                    // Create job objects for tracking
                    jobIds.forEach((jobId, sampleIdx) => {
                        newJobs.push({
                            id: jobId,
                            batchId: batchId,
                            sampleIndex: sampleIdx,
                            fileName: file.name,
                            status: 'queued',
                            result: null,
                        });

                        // Add prediction for first sample of this batch
                        if (sampleIdx === 0) {
                            addPrediction({
                                jobId: batchId,
                                caseId: selectedCase,
                                fileName: file.name,
                                status: 'queued',
                                numSamples: numSamples,
                            });
                        }
                    });

                    // Track batch info for display
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
        }
    };

    // Remove a selected file
    const removeFile = (index) => {
        setUploadedFiles(prev => prev.filter((_, i) => i !== index));
    };

    // Validate case form
    const validateCaseForm = () => {
        if (!newCaseForm.title.trim()) {
            setError('Case title is required');
            return false;
        }
        if (!newCaseForm.person_name.trim()) {
            setError('Person name is required');
            return false;
        }
        return true;
    };

    // Handle Enter key on case form inputs
    const handleCaseFormKeyDown = (e) => {
        if (e.key === 'Enter') {
            e.preventDefault();
            handleCreateCase(e);
        }
    };

    // Reset the entire analysis workflow
    const handleReset = () => {
        setStep('select-case');
        setSelectedCase(null);
        setUploadedFiles([]);
        setUploadedBatches([]);
        setProcessingJobs([]);
        setCompletedJobs([]);
        setTotalSamplesCount(0);
        setError(null);
        setIsUploading(false);
        setUploadProgress(0);
    };

    // Clear error message
    const clearError = () => {
        setError(null);
    };

    // Get selected case data
    const selectedCaseData = cases.find(c => c.id === selectedCase);

    return (
        <div className="w-full p-8">
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

            {/* Step 1: Select Case */}
            {step === 'select-case' && (
                <div>
                    <h2 className="text-2xl font-bold mb-2">Step 1: Select or Create Case</h2>
                    <p className="text-gray-600 mb-6">Choose an existing case or create a new one</p>

                    {isCreatingCase ? (
                        <div className="mb-8 p-6 bg-gradient-to-br from-blue-50 to-indigo-50 border border-blue-200 rounded-lg">
                            <h3 className="font-semibold text-gray-900 mb-4">Create New Case</h3>
                            <div className="space-y-4">
                                <div>
                                    <label className="block text-sm font-medium text-gray-700 mb-1">Title</label>
                                    <input
                                        type="text"
                                        placeholder="Case title"
                                        value={newCaseForm.title}
                                        onChange={(e) =>
                                            setNewCaseForm((prev) => ({ ...prev, title: e.target.value }))
                                        }
                                        onKeyDown={handleCaseFormKeyDown}
                                        className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                                    />
                                </div>
                                <div>
                                    <label className="block text-sm font-medium text-gray-700 mb-1">Description</label>
                                    <input
                                        type="text"
                                        placeholder="Case description"
                                        value={newCaseForm.description}
                                        onChange={(e) =>
                                            setNewCaseForm((prev) => ({ ...prev, description: e.target.value }))
                                        }
                                        onKeyDown={handleCaseFormKeyDown}
                                        className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                                    />
                                </div>
                                <div>
                                    <label className="block text-sm font-medium text-gray-700 mb-1">Subject Name</label>
                                    <input
                                        type="text"
                                        placeholder="Person name"
                                        value={newCaseForm.person_name}
                                        onChange={(e) =>
                                            setNewCaseForm((prev) => ({ ...prev, person_name: e.target.value }))
                                        }
                                        onKeyDown={handleCaseFormKeyDown}
                                        className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                                    />
                                </div>
                                <div className="flex gap-3 pt-2">
                                    <button
                                        onClick={handleCreateCase}
                                        className="flex-1 px-4 py-2 bg-blue-600 text-white font-medium rounded-lg hover:bg-blue-700 transition"
                                    >
                                        Create Case
                                    </button>
                                    <button
                                        onClick={() => setIsCreatingCase(false)}
                                        className="flex-1 px-4 py-2 bg-gray-300 text-gray-800 font-medium rounded-lg hover:bg-gray-400 transition"
                                    >
                                        Cancel
                                    </button>
                                </div>
                            </div>
                        </div>
                    ) : (
                        <button
                            onClick={() => setIsCreatingCase(true)}
                            className="mb-8 px-4 py-2 bg-blue-600 text-white font-medium rounded-lg hover:bg-blue-700 transition"
                        >
                            + Create New Case
                        </button>
                    )}

                    {cases.length > 0 ? (
                        <>
                            <h3 className="text-sm font-semibold text-gray-700 uppercase mb-4">Existing Cases ({cases.length})</h3>
                            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3 mb-6">
                                {cases.map((caseItem) => (
                                    <div
                                        key={caseItem.id}
                                        onClick={() => setSelectedCase(caseItem.id)}
                                        className={`p-4 border-2 rounded-lg cursor-pointer transition ${
                                            selectedCase === caseItem.id
                                                ? 'bg-blue-50 border-blue-500 shadow-md'
                                                : 'bg-white border-gray-200 hover:border-blue-300 hover:shadow-sm'
                                        }`}
                                    >
                                        <p className="font-semibold text-gray-900">
                                            {caseItem.title || 'Untitled'}
                                        </p>
                                        {caseItem.description && (
                                            <p className="text-xs text-gray-600 mt-1 truncate">
                                                {caseItem.description}
                                            </p>
                                        )}
                                        <p className="text-xs text-gray-500 mt-2">
                                            Subject: <span className="font-medium text-gray-700">{caseItem.person_name || 'Unknown'}</span>
                                        </p>
                                        <p className="text-xs text-gray-500 mt-1">
                                            Samples: <span className="font-bold text-blue-600">{caseItem.job_ids?.length || 0}</span>
                                        </p>
                                    </div>
                                ))}
                            </div>
                        </>
                    ) : (
                        <div className="mb-6 p-4 bg-gray-50 border border-gray-200 rounded-lg text-center">
                            <p className="text-gray-600">No existing cases. Create a new one to get started.</p>
                        </div>
                    )}

                    <button
                        onClick={handleNextStep}
                        disabled={!selectedCase}
                        className="w-full px-6 py-3 bg-green-600 text-white font-medium rounded-lg hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed transition"
                    >
                        Next: Upload Files →
                    </button>
                </div>
            )}

            {/* Step 2: Upload Files */}
            {step === 'upload-files' && (
                <div>
                    {!isUploading ? (
                        <>
                            <FileUploader
                                uploadedFiles={uploadedFiles}
                                onFilesChange={setUploadedFiles}
                                onRemoveFile={removeFile}
                                caseTitle={selectedCaseData?.title}
                            />

                            <div className="flex gap-2 mt-6">
                                <button
                                    onClick={handleNextStep}
                                    disabled={uploadedFiles.length === 0}
                                    className="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed"
                                >
                                    Next: Process Files
                                </button>
                                <button
                                    onClick={() => {
                                        setStep('select-case');
                                        setUploadedFiles([]);
                                    }}
                                    className="px-4 py-2 bg-gray-400 text-white rounded hover:bg-gray-500"
                                >
                                    Back
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

            {/* Step 3: Processing */}
            {step === 'processing' && (
                <div>
                    <h2 className="text-2xl font-bold mb-2">Step 3: Processing</h2>
                    <p className="text-gray-600 mb-6">Your samples are being analyzed</p>

                    {/* Overall Progress Summary */}
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

                                    {/* Overall Progress Bar */}
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

                    {/* Batches */}
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

                                    {/* Batch Progress Bar */}
                                    <div className="mb-4">
                                        <div className="w-full bg-gray-200 rounded-full h-2 overflow-hidden">
                                            <div
                                                className="h-2 rounded-full bg-gradient-to-r from-blue-400 to-blue-600 transition-all duration-300"
                                                style={{ width: `${batchProgress}%` }}
                                            />
                                        </div>
                                    </div>

                                    {/* Sample Status Grid */}
                                    <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-2">
                                        {batchJobs.map((job, sampleIdx) => {
                                            const isCompleted = job.status === 'completed';
                                            const isRunning = job.status === 'running';
                                            const isQueued = job.status === 'queued';
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
                                                    {isRunning && (
                                                        <p className="text-xs mt-1">Processing...</p>
                                                    )}
                                                    {isError && (
                                                        <p className="text-xs mt-1">✕ Error</p>
                                                    )}
                                                    {isQueued && (
                                                        <p className="text-xs mt-1">Waiting...</p>
                                                    )}
                                                </div>
                                            );
                                        })}
                                    </div>
                                </div>
                            );
                        })}
                    </div>

                    {/* Info Message */}
                    <div className="mt-8 p-4 bg-blue-50 border border-blue-200 rounded-lg">
                        <p className="text-sm text-blue-800">
                            <strong>Processing in progress...</strong> You'll be automatically taken to results when all samples complete.
                        </p>
                    </div>
                </div>
            )}

            {/* Step 4: Results */}
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