# CSV Filename Preservation Fix

## Problem
When uploading CSV files, all files were showing up as "sample_converted.csv" in the case detail view instead of preserving their original filenames.

## Root Cause
1. **Backend (jobs.py line 108)**: Original uploaded files were saved with hardcoded name `sample.csv`
2. **Backend (jobs.py line 134)**: Converted files were saved with hardcoded name `sample_converted.csv`
3. **Backend (jobs.py line 162)**: The `file_path` stored in the Job model pointed to `sample_converted.csv`
4. **Database (models.py)**: No column existed to store the original filename
5. **Frontend (CaseDetailPage.jsx line 48)**: Filename was extracted from `file_path` using `.split('/').pop()`, which always returned `sample_converted.csv`

## Solution

### 1. Database Schema Change
**File**: `/Users/connorjason/VSCProjects/SharkMQP26/backend/app/models.py`

Added new column to Job model:
```python
original_filename = Column(String(255), nullable=True)
```

Updated `Job.to_dict()` method to include the new field in API responses.

### 2. Backend Upload Logic
**File**: `/Users/connorjason/VSCProjects/SharkMQP26/backend/app/jobs.py`

Modified `upload_and_enqueue` endpoint:
- Capture original filename from `file.filename` (line ~92)
- Store it in the Job record when creating jobs (line ~163)

Modified rerun endpoints:
- `rerun_job` (line ~377): Copy original_filename from original job
- `rerun_batch` (line ~472): Copy original_filename for all jobs in batch

### 3. Frontend Display Logic

**File**: `/Users/connorjason/VSCProjects/SharkMQP26/frontend/src/pages/CaseDetailPage.jsx`

Updated batch reconstruction logic (line ~48):
- Use `job.original_filename` first
- Fallback to extracting from `file_path` for legacy data
- Fallback to 'Unknown' if neither is available

```javascript
fileName: job.original_filename || job.file_path?.split('/').pop() || 'Unknown'
```

**File**: `/Users/connorjason/VSCProjects/SharkMQP26/frontend/src/pages/CasePage.jsx`

Updated job display logic (line ~301):
```javascript
{job.original_filename || (job.file_path ? job.file_path.split('/').pop() : 'Unknown file')}
```

### 4. Database Migration
**File**: `/Users/connorjason/VSCProjects/SharkMQP26/backend/migrations/001_add_original_filename.py`

Created migration script to:
- Add `original_filename` column to existing databases
- Populate existing records with fallback filename "legacy_upload.csv"
- Support rollback functionality

## Running the Migration

For existing databases, run:
```bash
python backend/migrations/001_add_original_filename.py
```

To rollback:
```bash
python backend/migrations/001_add_original_filename.py --rollback
```

## Testing

### New Uploads
1. Upload a CSV file with any name (e.g., "patient_data_2025.csv")
2. Navigate to the case detail page
3. Verify the filename displays as "patient_data_2025.csv"

### Existing Data
- Legacy records will show as "legacy_upload.csv" after migration
- The actual data and results are unaffected

### Multiple Files
1. Upload multiple CSV files with different names
2. Verify each file displays its correct original filename
3. Verify samples are correctly grouped by batch

## Files Changed

### Backend
- `/Users/connorjason/VSCProjects/SharkMQP26/backend/app/models.py` - Added original_filename column
- `/Users/connorjason/VSCProjects/SharkMQP26/backend/app/jobs.py` - Store and copy original_filename

### Frontend
- `/Users/connorjason/VSCProjects/SharkMQP26/frontend/src/pages/CaseDetailPage.jsx` - Use original_filename for display in case detail view
- `/Users/connorjason/VSCProjects/SharkMQP26/frontend/src/pages/CasePage.jsx` - Use original_filename for display in cases list

### New Files
- `/Users/connorjason/VSCProjects/SharkMQP26/backend/migrations/001_add_original_filename.py` - Migration script
- `/Users/connorjason/VSCProjects/SharkMQP26/backend/migrations/README.md` - Migration documentation
- `/Users/connorjason/VSCProjects/SharkMQP26/backend/test_filename_fix.py` - Test script to validate the fix
- `/Users/connorjason/VSCProjects/SharkMQP26/FILENAME_FIX_SUMMARY.md` - This document

## Impact
- Backward compatible: Old code will continue to work
- Frontend gracefully handles missing original_filename (falls back to file_path extraction)
- Migration script handles existing databases
- No breaking changes to API contracts

## Validation

Run the test script to verify everything is working:
```bash
python backend/test_filename_fix.py
```

This will check:
- Database column exists
- Model has the attribute
- to_dict() includes the field
- Queries work correctly

## Next Steps
1. Run the migration script on production database: `python backend/migrations/001_add_original_filename.py`
2. Run the test script to validate: `python backend/test_filename_fix.py`
3. Test uploading new files to verify filenames are preserved
4. Monitor logs for any issues with the new column
