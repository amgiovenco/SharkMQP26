import { useState } from 'react';
import { apiFetch } from '../../utility/ApiFetch';
import { useCasesStore } from '../../stores/casesStore';

const CaseSelector = ({ onCaseSelected }) => {
  const { cases, addCase } = useCasesStore();
  const [selectedCase, setSelectedCase] = useState(null);
  const [isCreatingCase, setIsCreatingCase] = useState(false);
  const [newCaseForm, setNewCaseForm] = useState({
    title: '',
    description: '',
    person_name: '',
  });
  const [error, setError] = useState(null);

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
        onCaseSelected(caseData.id);
      })
      .catch(err => {
        setError(`Failed to create case: ${err.message}`);
      });
  };

  const handleSelectCase = (caseId) => {
    setSelectedCase(caseId);
    onCaseSelected(caseId);
  };

  return (
    <div>
      <h2 className="text-xl font-semibold mb-4">Step 1: Select or Create Case</h2>

      {error && (
        <div className="mb-4 p-3 border border-red-300 bg-red-50">
          <p className="text-red-800 text-sm">{error}</p>
        </div>
      )}

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
            onClick={() => handleSelectCase(caseItem.id)}
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

      {selectedCase && (
        <p className="text-sm text-gray-600">
          Selected: <strong>{cases.find(c => c.id === selectedCase)?.title}</strong>
        </p>
      )}
    </div>
  );
};

export default CaseSelector;
