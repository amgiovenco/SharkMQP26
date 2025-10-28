import { useEffect } from 'react';
import { useSocket } from '../../contexts/SocketContext';

export const useJobStatusListener = (processingJobs, setProcessingJobs, setCompletedJobs, onComplete) => {
    const socket = useSocket();

    useEffect(() => {
        if (!socket || processingJobs.length === 0) return;

        // Handler for job status updates
        const handleJobStatus = (data) => {
            const { job_id, status, result, error } = data;

            // Update job status in processingJobs
            setProcessingJobs(prev => {
                const job = prev.find(j => j.id === job_id);
                if (!job) return prev;

                if (status === 'completed' || status === 'failed') {
                    const completedJob = { ...job, status, result, error };
                    setCompletedJobs(prevCompleted => [...prevCompleted, completedJob]);
                    return prev.filter(j => j.id !== job_id);
                }

                return prev.map(j => j.id === job_id ? { ...j, status } : j);
            });

            // Check if all jobs are done
            setProcessingJobs(prev => {
                const remaining = prev.filter(j => j.id !== job_id && (j.status === 'queued' || j.status === 'processing'));
                if (remaining.length === 0 && prev.some(j => j.id === job_id)) {
                    onComplete();
                }
                return remaining;
            });
        };

        socket.on('job_status', handleJobStatus);

        return () => socket.off('job_status', handleJobStatus);

    }, [socket, processingJobs, setProcessingJobs, setCompletedJobs, onComplete]);
};