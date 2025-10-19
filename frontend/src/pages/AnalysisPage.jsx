// when you upload csv and then wait for resutls and then see results
import { useState, useRef } from 'react';
import { apiFetch } from '../utility/ApiFetch';
import { useCasesStore } from '../stores/casesStore';
import { usePredictionStore } from '../stores/predictionStore';
import { useJobStatusListener } from '../components/analysis/useJobStatusListener';

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
  const fileInputRef = useRef(null);

  useJobStatusListener(
        processingJobs,
        setProcessingJobs,
        setCompletedJobs,
        () => setStep('results')
    );

  const handleCreateCase = (e) => {
    e.preventDefault();
    apiFetch('/cases', {
      method: 'POST',
      body: JSON.stringify(newCaseForm),
    })
      .then(caseData => {
        addCase(caseData);
        setSelectedCase(caseData.id);
        setNewCaseForm({ title: '', description: '', person_name: '' });
        setIsCreatingCase(false);
      })
      .catch(err => {
        setError(`Failed to create case: ${err.message}`);
      });
  };

  const handleNextStep = () => {
    if (step === 'select-case' && selectedCase) {
      setStep('upload-files');
      setError(null);
    } else if (step === 'upload-files' && uploadedFiles.length > 0) {
      // Upload all files
      setError(null);
      setProcessingJobs([]);
      setCompletedJobs([]);

      const newJobs = [];
      let uploadCount = 0;

      uploadedFiles.forEach(file => {
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
            addPrediction({
              jobId: jobData.job.id,
              caseId: selectedCase,
              fileName: file.name,
              status: jobData.status,
              createdAt: jobData.job.created_at,
            });

            if (uploadCount === uploadedFiles.length) {
              setProcessingJobs(newJobs);
              setStep('processing');
            }
          })
          .catch(err => {
            setError(`Upload error: ${err.message}`);
          });
      });
    }
  };

  const handleFileSelect = (e) => {
    const files = Array.from(e.target.files || []);
    setUploadedFiles(prev => [...prev, ...files]);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const removeFile = (index) => {
    setUploadedFiles(prev => prev.filter((_, i) => i !== index));
  };

  const handleReset = () => {
    setStep('select-case');
    setSelectedCase(null);
    setUploadedFiles([]);
    setProcessingJobs([]);
    setCompletedJobs([]);
    setError(null);
  };

  const selectedCaseData = cases.find(c => c.id === selectedCase);

  return (
    <div className="p-8 max-w-2xl">
      <h1 className="text-3xl font-bold mb-8">Analysis</h1>

      {error && (
        <div className="mb-6 p-4 border border-red-300 bg-red-50">
          <p className="text-red-800">{error}</p>
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
                className="w-full p-2 border rounded"
              />
              <input
                type="text"
                placeholder="Description"
                value={newCaseForm.description}
                onChange={(e) =>
                  setNewCaseForm((prev) => ({ ...prev, description: e.target.value }))
                }
                className="w-full p-2 border rounded"
              />
              <input
                type="text"
                placeholder="Person Name"
                value={newCaseForm.person_name}
                onChange={(e) =>
                  setNewCaseForm((prev) => ({ ...prev, person_name: e.target.value }))
                }
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
                <p className="font-medium">{caseItem.title || 'Untitled'}</p>
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
          <h2 className="text-xl font-semibold mb-4">Step 2: Upload Files</h2>
          <p className="text-sm text-gray-600 mb-4">
            Case: <strong>{selectedCaseData?.title}</strong>
          </p>

          <div
            onClick={() => fileInputRef.current?.click()}
            className="border-2 border-dashed rounded p-8 text-center cursor-pointer hover:bg-gray-50 transition mb-6"
          >
            <p className="font-medium mb-2">Click to select CSV files</p>
            <p className="text-sm text-gray-600">You can select multiple files</p>
          </div>

          <input
            ref={fileInputRef}
            type="file"
            accept=".csv"
            onChange={handleFileSelect}
            multiple
            hidden
          />

          {uploadedFiles.length > 0 && (
            <div className="mb-6">
              <h3 className="font-medium mb-2">Selected Files ({uploadedFiles.length})</h3>
              <div className="space-y-2">
                {uploadedFiles.map((file, i) => (
                  <div key={i} className="flex justify-between items-center p-2 border rounded bg-gray-50">
                    <span className="text-sm">{file.name}</span>
                    <button
                      onClick={() => removeFile(i)}
                      className="px-2 py-1 text-sm bg-red-500 text-white rounded hover:bg-red-600"
                    >
                      Remove
                    </button>
                  </div>
                ))}
              </div>
            </div>
          )}

          <div className="flex gap-2">
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
            onClick={handleReset}
            className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
          >
            Start New Analysis
          </button>
        </div>
      )}
    </div>
  );
};

export default AnalysisPage;