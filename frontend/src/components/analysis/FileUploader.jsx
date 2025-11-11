import { useState, useRef } from 'react';

// Component for uploading files associated with a case
const FileUploader = ({ caseTitle }) => {
    const [uploadedFiles, setUploadedFiles] = useState([]);
    const fileInputRef = useRef(null);

    // Handle file selection
    const handleFileSelect = (e) => {
        const files = Array.from(e.target.files || []);
        setUploadedFiles(prev => [...prev, ...files]);
        if (fileInputRef.current) fileInputRef.current.value = '';
    };

    // Remove a file from the list
    const removeFile = (index) => {
        setUploadedFiles(prev => prev.filter((_, i) => i !== index));
    };

    return (
        <div>
            <h2 className="text-xl font-semibold mb-4">Step 2: Upload Files</h2>
            <p className="text-sm text-gray-600 mb-4">
                Case: <strong>{caseTitle}</strong>
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

            {uploadedFiles}
        </div>
    );
};

export default FileUploader;
