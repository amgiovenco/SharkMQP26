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
    const [processingJobs, setProcessingJobs] = useState([]);
    const [completedJobs, setCompletedJobs] = useState([]);
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

            const newJobs = [];
            let uploadCount = 0;
            const failedUploads = [];

            uploadedFiles.forEach((file, index) => {
                const formData = new FormData();
                formData.append('file', file);
                formData.append('case_id', selectedCase);

                apiFetch('/jobs/upload', {
                    method: 'POST',
                    body: formData,
                })
                .then(jobData => {
                    newJobs.push(jobData.job);
                    uploadCount++;
                    setUploadProgress(Math.round((uploadCount / uploadedFiles.length) * 100));
                    addPrediction({
                        jobId: jobData.job.id,
                        caseId: selectedCase,
                        fileName: file.name,
                        status: jobData.status,
                        createdAt: jobData.job.created_at,
                    });

                    if (uploadCount === uploadedFiles.length) {
                        if (failedUploads.length > 0) {
                            setError(`${failedUploads.length} file(s) failed to upload. Proceeding with ${newJobs.length} successful uploads.`);
                        }
                        setIsUploading(false);
                        setProcessingJobs(newJobs);
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
        setProcessingJobs([]);
        setCompletedJobs([]);
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
        <div className="p-8 max-w-2xl">
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
                    <h2 className="text-xl font-semibold mb-4">Step 1: Select or Create Case</h2>

                    {isCreatingCase ? (
                        <div className="space-y-3 mb-6 p-4 border rounded">
                            <input
                                type="text"
                                placeholder="Title"
                                value={newCaseForm.title}
                                onChange={(e) =>
                                    setNewCaseForm((prev) => ({ ...prev, title: e.target.value }))
                                }
                                onKeyDown={handleCaseFormKeyDown}
                                className="w-full p-2 border rounded"
                            />
                            <input
                                type="text"
                                placeholder="Description"
                                value={newCaseForm.description}
                                onChange={(e) =>
                                    setNewCaseForm((prev) => ({ ...prev, description: e.target.value }))
                                }
                                onKeyDown={handleCaseFormKeyDown}
                                className="w-full p-2 border rounded"
                            />
                            <input
                                type="text"
                                placeholder="Person Name"
                                value={newCaseForm.person_name}
                                onChange={(e) =>
                                    setNewCaseForm((prev) => ({ ...prev, person_name: e.target.value }))
                                }
                                onKeyDown={handleCaseFormKeyDown}
                                className="w-full p-2 border rounded"
                            />
                            <div className="flex gap-2">
                                <button
                                    onClick={handleCreateCase}
                                    className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
                                >
                                    Create Case
                                </button>
                                <button
                                    onClick={() => setIsCreatingCase(false)}
                                    className="px-4 py-2 bg-gray-400 text-white rounded hover:bg-gray-500"
                                >
                                    Cancel
                                </button>
                            </div>
                        </div>
                    ) : (
                        <button
                            onClick={() => setIsCreatingCase(true)}
                            className="mb-6 px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
                        >
                            + New Case
                        </button>
                    )}

                    <div className="space-y-2 mb-6">
                        {cases.map((caseItem) => (
                            <div
                                key={caseItem.id}
                                onClick={() => setSelectedCase(caseItem.id)}
                                className={`p-3 border rounded cursor-pointer transition ${
                                    selectedCase === caseItem.id
                                        ? 'bg-blue-100 border-blue-500'
                                        : 'hover:bg-gray-50'
                                }`}
                            >
                                <p className="font-medium">
                                    {caseItem.title || 'Untitled'}
                                </p>
                                <p className="text-sm text-gray-600">
                                    {caseItem.person_name || 'No subject'}
                                </p>
                            </div>
                        ))}
                    </div>

                    <button
                        onClick={handleNextStep}
                        disabled={!selectedCase}
                        className="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                        Next: Upload Files
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
                    <h2 className="text-xl font-semibold mb-4">Step 3: Processing</h2>
                    <p className="text-sm text-gray-600 mb-6">
                        Processing {uploadedFiles.length} file(s)...
                    </p>

                    <div className="space-y-3">
                        {processingJobs.map((job, idx) => (
                            <div key={job.id} className="p-4 border rounded bg-blue-50">
                                <div className="flex justify-between items-center">
                                    <span className="text-sm font-medium">
                                        {uploadedFiles[idx]?.name}
                                    </span>
                                    <span className="text-sm text-gray-600">{job.status}</span>
                                </div>
                                <div className="w-full bg-gray-300 rounded h-2 mt-2">
                                    <div className="bg-blue-600 h-2 rounded w-1/2 animate-pulse" />
                                </div>
                            </div>
                        ))}

                        {completedJobs.length > 0 && (
                            <div className="mt-4 p-4 border rounded bg-green-50">
                                <p className="text-sm font-medium text-green-800">
                                    {completedJobs.length} job(s) completed
                                </p>
                            </div>
                        )}
                    </div>
                </div>
            )}

            {/* Step 4: Results */}
            {step === 'results' && (
                <AnalysisResults
                    completedJobs={completedJobs}
                    uploadedFiles={uploadedFiles}
                    onReset={handleReset}
                />
            )}
        </div>
    );
};

export default AnalysisPage;