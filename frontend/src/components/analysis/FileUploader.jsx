import { useRef } from 'react';

/**
 * FileUploader Component
 * Reusable component for uploading CSV files
 *
 * @param {File[]} uploadedFiles - Array of currently selected files
 * @param {Function} onFilesChange - Callback when files are added (receives File[])
 * @param {Function} onRemoveFile - Callback to remove a file by index
 * @param {string} caseTitle - Title of the case (for display)
 */
const FileUploader = ({ uploadedFiles, onFilesChange, onRemoveFile, caseTitle }) => {
    const fileInputRef = useRef(null);

    // Handle file selection
    const handleFileSelect = (e) => {
        const files = Array.from(e.target.files || []);
        if (files.length > 0) {
            onFilesChange([...uploadedFiles, ...files]);
        }
        if (fileInputRef.current) fileInputRef.current.value = '';
    };

    const handleRemoveFile = (index) => {
        onRemoveFile(index);
    };

    return (
        <div>
            <h2 className="text-xl font-semibold mb-4">Step 2: Upload Files</h2>
            {caseTitle && (
                <p className="text-sm text-gray-600 mb-4">
                    Case: <strong>{caseTitle}</strong>
                </p>
            )}

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
                                    onClick={() => handleRemoveFile(i)}
                                    className="px-2 py-1 text-sm bg-red-500 text-white rounded hover:bg-red-600"
                                >
                                    Remove
                                </button>
                            </div>
                        ))}
                    </div>
                </div>
            )}
        </div>
    );
};

export default FileUploader;
