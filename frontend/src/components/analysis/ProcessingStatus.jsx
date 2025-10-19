const ProcessingStatus = ({ processingJobs, completedJobs, uploadedFiles }) => {
  return (
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
  );
};

export default ProcessingStatus;
