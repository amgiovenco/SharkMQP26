// when u click 'view' on the analysis history page
import { useParams, useNavigate } from 'react-router-dom';
import { useEffect, useState } from 'react';
import { apiFetch } from '../utility/ApiFetch';

const CasePage = () => {
  const { caseId } = useParams();
  const navigate = useNavigate();
  const [caseItem, setCaseItem] = useState(null);
  const [jobs, setJobs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
        const fetchCaseData = async () => {
            setLoading(true);
            setError(null);

            try {
                // Fetch case details
                const caseData = await apiFetch(`/cases/${caseId}`);
                setCaseItem(caseData);

                // Fetch jobs for this case
                const jobsData = await apiFetch(`/cases/${caseId}/jobs`);
                setJobs(jobsData.jobs || []);
            } catch (err) {
                setError(err.message);
            } finally {
                setLoading(false);
            }
        };

        fetchCaseData();
  }, [caseId]);

  if (loading) {
        return (
        <div className="p-8">
            <div>Loading case...</div>
        </div>
        );
  }

  if (error) {
        return (
        <div className="p-8">
            <div>Error: {error}</div>
        </div>
        );
  }

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

                <div>
                    <h2 className="text-lg font-semibold mb-3">Analyses ({jobs.length})</h2>
                    {jobs.length === 0 ? (
                        <p className="text-gray-500">No analyses yet</p>
                    ) : (
                        <div className="space-y-2">
                            {jobs.map((job) => (
                                <div key={job.id} className="p-3 border rounded">
                                    <div className="flex justify-between items-start">
                                        <div className="flex-1">
                                            <p className="font-medium">{job.status}</p>
                                            <p className="text-sm text-gray-600">
                                                {new Date(job.created_at).toLocaleString()}
                                            </p>
                                        </div>
                                        <p className="text-xs text-gray-500">
                                            {job.sha256?.slice(0, 8)}...
                                        </p>
                                    </div>
                                </div>
                            ))}
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

export default CasePage;