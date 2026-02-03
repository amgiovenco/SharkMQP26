// when u click 'view' on the analysis history page
import { useParams, useNavigate } from 'react-router-dom';
import { useEffect, useState, useRef } from 'react';
import { apiFetch } from '../utility/ApiFetch';
import { useSocket } from '../contexts/SocketContext';

const CasePage = () => {
    const { caseId } = useParams();
    const navigate = useNavigate();
    const fileInputRef = useRef(null);
    const [caseItem, setCaseItem] = useState(null);
    const [jobs, setJobs] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [isUploading, setIsUploading] = useState(false);
    const [uploadProgress, setUploadProgress] = useState(0);
    const [jobsPage, setJobsPage] = useState(1);
    const [jobsTotal, setJobsTotal] = useState(0);
    const jobsPerPage = 20;
    const socket = useSocket();

    const fetchCaseData = async (page = 1) => {
        setLoading(true);
        setError(null);

        try {
            // Fetch case details
            const caseData = await apiFetch(`/cases/${caseId}`);
            setCaseItem(caseData);

            // Fetch jobs for this case with pagination
            const jobsData = await apiFetch(`/cases/${caseId}/jobs?page=${page}&per_page=${jobsPerPage}`);
            setJobs(jobsData.jobs || []);
            setJobsTotal(jobsData.total || 0);
            setJobsPage(page);
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        fetchCaseData(1);
    }, [caseId]);

    // Listen for real-time job status updates via socket
    useEffect(() => {
        if (!socket) {
            console.log('CasePage: Socket not ready');
            return;
        }

        console.log('CasePage: Setting up socket listener, currently have', jobs.length, 'jobs');

        const handleJobStatus = (data) => {
            console.log('CasePage: Received job_status event:', data);
            const { job_id, status, result, error } = data;

            // Update the job in the jobs list if it belongs to this case
            setJobs(prevJobs => {
                const jobIndex = prevJobs.findIndex(j => j.id === job_id);
                console.log('CasePage: Looking for job', job_id, '- found at index:', jobIndex, 'in', prevJobs.length, 'jobs');

                if (jobIndex === -1) {
                    console.log('CasePage: Job not in current page, skipping update');
                    return prevJobs; // Job not in current list
                }

                console.log('CasePage: Updating job at index', jobIndex, 'with status:', status);

                const updatedJob = {
                    ...prevJobs[jobIndex],
                    status,
                    result_json: result || prevJobs[jobIndex].result_json,
                    error: error || prevJobs[jobIndex].error,
                };

                // Update finished_at if job is now complete/failed
                if ((status === 'completed' || status === 'failed' || status === 'done' || status === 'error') && !prevJobs[jobIndex].finished_at) {
                    updatedJob.finished_at = new Date().toISOString();
                }

                // Update started_at if job is now running and hasn't started
                if ((status === 'running' || status === 'processing') && !prevJobs[jobIndex].started_at) {
                    updatedJob.started_at = new Date().toISOString();
                }

                const newJobs = [...prevJobs];
                newJobs[jobIndex] = updatedJob;
                console.log('CasePage: Job updated, new status:', updatedJob.status);
                return newJobs;
            });
        };

        console.log('CasePage: Attaching job_status listener');
        socket.on('job_status', handleJobStatus);

        return () => {
            console.log('CasePage: Removing job_status listener');
            socket.off('job_status', handleJobStatus);
        };
    }, [socket, jobs.length]);

    // Handle file upload to this case
    const handleFileSelect = async (event) => {
        const files = Array.from(event.target.files || []);
        if (files.length === 0) return;

        setIsUploading(true);
        setError(null);
        setUploadProgress(0);

        let uploadedCount = 0;
        const failedFiles = [];

        for (const file of files) {
            try {
                const formData = new FormData();
                formData.append('file', file);
                formData.append('case_id', caseId);

                await apiFetch('/jobs/upload', {
                    method: 'POST',
                    body: formData,
                });

                uploadedCount++;
                setUploadProgress(Math.round((uploadedCount / files.length) * 100));
            } catch (err) {
                failedFiles.push(file.name);
            }
        }

        if (failedFiles.length > 0) {
            setError(`Failed to upload ${failedFiles.length} file(s): ${failedFiles.join(', ')}`);
        } else {
            // Refresh jobs after successful upload
            await fetchCaseData(1);
            setUploadProgress(0);
        }

        setIsUploading(false);
        // Reset file input
        if (fileInputRef.current) {
            fileInputRef.current.value = '';
        }
    };

    // Helper functions
    const getStatusColor = (status) => {
        switch (status) {
            case 'done':
            case 'completed':
                return 'bg-green-100 text-green-800';
            case 'running':
            case 'processing':
                return 'bg-blue-100 text-blue-800';
            case 'queued':
            case 'pending':
                return 'bg-yellow-100 text-yellow-800';
            case 'error':
            case 'failed':
                return 'bg-red-100 text-red-800';
            default:
                return 'bg-gray-100 text-gray-800';
        }
    };

    const getStatusLabel = (status) => {
        const statusMap = {
            'done': 'Completed',
            'completed': 'Completed',
            'running': 'Processing',
            'processing': 'Processing',
            'queued': 'Queued',
            'pending': 'Pending',
            'error': 'Error',
            'failed': 'Error',
        };
        return statusMap[status] || status;
    };

    const calculateDuration = (startedAt, finishedAt) => {
        if (!startedAt || !finishedAt) return null;
        const start = new Date(startedAt);
        const finish = new Date(finishedAt);
        const seconds = Math.round((finish - start) / 1000);
        return seconds > 60 ? `${(seconds / 60).toFixed(1)}m` : `${seconds}s`;
    };

    const calculateWaitTime = (createdAt, startedAt) => {
        if (!createdAt || !startedAt) return null;
        const created = new Date(createdAt);
        const started = new Date(startedAt);
        const seconds = Math.round((started - created) / 1000);
        return seconds > 60 ? `${(seconds / 60).toFixed(1)}m` : `${seconds}s`;
    };

    // Loading state
    if (loading) {
        return (
            <div className="p-8">
                <div>Loading case...</div>
            </div>
        );
    }

    // Error state
    if (error) {
        return (
            <div className="p-8">
                <div>Error: {error}</div>
            </div>
        );
    }

    // Case not found
    if (!caseItem) {
        return (
            <div className="p-8">
                <div>Case not found</div>
            </div>
        );
    }

    return (
        <div className="p-8">
            <button 
                onClick={() => navigate('/history')}
                className="mb-6 px-4 py-2 border rounded hover:bg-gray-100"
            >
                ← Back to History
            </button>

            <div className="mb-8">
                <h1 className="text-3xl font-bold mb-4">
                    {caseItem.title || 'Untitled Case'}
                </h1>
                
                <div className="space-y-3 mb-6">
                    <div>
                        <p className="text-sm text-gray-600">Description</p>
                        <p>{caseItem.description || 'No description'}</p>
                    </div>
                    
                    <div>
                        <p className="text-sm text-gray-600">Subject</p>
                        <p>{caseItem.person_name || 'Unknown'}</p>
                    </div>
                    
                    <div>
                        <p className="text-sm text-gray-600">Created</p>
                        <p>{new Date(caseItem.created_at).toLocaleString()}</p>
                    </div>
                </div>

                <div className="my-6 p-4 bg-blue-50 border border-blue-200 rounded-lg">
                    <h3 className="font-semibold text-blue-900 mb-2">Add More Analyses</h3>
                    <p className="text-sm text-blue-800 mb-3">Upload CSV files to run more analyses on this case</p>
                    <div className="flex gap-2 items-center">
                        <input
                            ref={fileInputRef}
                            type="file"
                            multiple
                            accept=".csv"
                            onChange={handleFileSelect}
                            disabled={isUploading}
                            className="hidden"
                        />
                        <button
                            onClick={() => fileInputRef.current?.click()}
                            disabled={isUploading}
                            className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
                        >
                            {isUploading ? `Uploading ${uploadProgress}%...` : 'Choose CSV Files'}
                        </button>
                    </div>
                </div>

                <div>
                    <h2 className="text-lg font-semibold mb-3">Analyses ({jobsTotal})</h2>
                    {jobs.length === 0 ? (
                        <p className="text-gray-500">No analyses yet</p>
                    ) : (
                        <>
                            <div className="space-y-3">
                                {jobs.map((job) => (
                                    <div key={job.id} className="p-4 border rounded-lg hover:bg-gray-50">
                                        <div className="flex justify-between items-start mb-2">
                                            <div className="flex-1">
                                                <div className="flex items-center gap-2 mb-2">
                                                    <span className={`px-2 py-1 rounded text-xs font-semibold ${getStatusColor(job.status)}`}>
                                                        {getStatusLabel(job.status)}
                                                    </span>
                                                    {job.user && (
                                                        <p className="text-xs text-gray-500">by {job.user.full_name}</p>
                                                    )}
                                                </div>
                                                <p className="font-medium text-sm mb-1">
                                                    {job.original_filename || (job.file_path ? job.file_path.split('/').pop() : 'Unknown file')}
                                                </p>
                                            </div>
                                            <p className="text-xs text-gray-400 font-mono">
                                                {job.sha256?.slice(0, 8)}
                                            </p>
                                        </div>

                                        <div className="grid grid-cols-2 gap-2 text-xs text-gray-600 mb-2">
                                            <div>
                                                <p className="text-gray-500">Submitted</p>
                                                <p>{new Date(job.created_at).toLocaleString()}</p>
                                            </div>
                                            {job.started_at && (
                                                <div>
                                                    <p className="text-gray-500">Started</p>
                                                    <p>{new Date(job.started_at).toLocaleString()}</p>
                                                </div>
                                            )}
                                        </div>

                                        {(calculateWaitTime(job.created_at, job.started_at) || calculateDuration(job.started_at, job.finished_at)) && (
                                            <div className="grid grid-cols-2 gap-2 text-xs text-gray-600 mb-2">
                                                {calculateWaitTime(job.created_at, job.started_at) && (
                                                    <div>
                                                        <p className="text-gray-500">Queue wait</p>
                                                        <p>{calculateWaitTime(job.created_at, job.started_at)}</p>
                                                    </div>
                                                )}
                                                {calculateDuration(job.started_at, job.finished_at) && (
                                                    <div>
                                                        <p className="text-gray-500">Processing time</p>
                                                        <p>{calculateDuration(job.started_at, job.finished_at)}</p>
                                                    </div>
                                                )}
                                            </div>
                                        )}

                                        {job.result_json && (
                                            <div className="bg-blue-50 p-2 rounded text-xs border border-blue-200">
                                                <p className="font-semibold text-blue-900 mb-1">Result Preview</p>
                                                <p className="text-blue-800">
                                                    Top: <strong>{job.result_json.winner || job.result_json.topk?.[0]?.label || 'N/A'}</strong>
                                                    {job.result_json.confidence && ` (${(job.result_json.confidence * 100).toFixed(1)}%)`}
                                                    {!job.result_json.confidence && job.result_json.topk?.[0]?.prob && ` (${(job.result_json.topk[0].prob * 100).toFixed(1)}%)`}
                                                </p>
                                            </div>
                                        )}

                                        {job.status === 'error' && job.error && (
                                            <div className="bg-red-50 p-2 rounded text-xs border border-red-200">
                                                <p className="text-red-800">{job.error}</p>
                                            </div>
                                        )}
                                    </div>
                                ))}
                            </div>

                            {jobsTotal > jobsPerPage && (
                                <div className="flex gap-2 mt-6 justify-center items-center">
                                    <button
                                        onClick={() => fetchCaseData(jobsPage - 1)}
                                        disabled={jobsPage === 1}
                                        className="px-3 py-1 border rounded disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-100"
                                    >
                                        ← Previous
                                    </button>
                                    <span className="text-sm text-gray-600">
                                        Page {jobsPage} of {Math.ceil(jobsTotal / jobsPerPage)}
                                    </span>
                                    <button
                                        onClick={() => fetchCaseData(jobsPage + 1)}
                                        disabled={jobsPage >= Math.ceil(jobsTotal / jobsPerPage)}
                                        className="px-3 py-1 border rounded disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-100"
                                    >
                                        Next →
                                    </button>
                                </div>
                            )}
                        </>
                    )}
                </div>
            </div>
        </div>
    );
};

export default CasePage;