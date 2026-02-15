import json
from datetime import datetime, timezone
from pathlib import Path
import tempfile

from redis.asyncio import from_url
from sqlalchemy.orm import Session
import asyncio
import torch

from app.db import SessionLocal
from app.models import Job, JobResult
from app.settings import settings
from app.logger import get_logger
from worker.cnn_inference import ml_inference, CNNModel
from worker.extract_melt_block import process_file as extract_melt_block

logger = get_logger(__name__)

# Determine device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def _preprocess_raw_csv(filepath: str) -> str:
    """
    Preprocess raw user input CSV if needed.

    If the file appears to be raw instrument output (has melt block markers),
    extract and clean it. Otherwise, return as-is.

    Args:
        filepath: Path to input CSV file

    Returns:
        Path to cleaned/processed CSV file
    """
    input_path = Path(filepath)

    # Check if file contains melt block markers (indicates raw instrument data)
    try:
        with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read(10000)  # Check first 10KB
            has_markers = ("Start Worksheet" in content or "End Worksheet" in content)
    except Exception as e:
        logger.warning(f"Could not check file markers: {e}, assuming cleaned CSV")
        return filepath

    if not has_markers:
        # Already cleaned, use as-is
        return filepath

    # Extract and clean the raw data
    logger.info(f"Detected raw instrument data, extracting melt block from {input_path.name}")

    try:
        # Create temp file for output
        _, temp_path = tempfile.mkstemp(suffix='.csv', prefix=f"{input_path.stem}_processed_")
        temp_file = Path(temp_path)

        # Extract melt block
        extract_melt_block(str(input_path), str(temp_file))

        logger.info(f"Successfully extracted melt block to {temp_file}")
        return str(temp_file)

    except Exception as e:
        logger.error(f"Failed to extract melt block: {e}")
        raise ValueError(f"Could not process raw instrument data: {e}")

def run_inference(filepath: str, sample_index: int = 0):
    """Run ML inference on a specific sample in the CSV file.

    Automatically preprocesses raw instrument data if needed.
    """
    try:
        # Preprocess raw CSV if necessary
        processed_filepath = _preprocess_raw_csv(filepath)
        logger.info(f"[run_inference] Using filepath: {processed_filepath}")

        # Run inference on processed data
        result = ml_inference(processed_filepath, sample_index=sample_index, device=DEVICE)

        # Log result structure
        logger.info(f"[run_inference] Got result, success={result['success']}")
        if not result['success']:
            error_msg = result.get('error', 'Unknown error')
            logger.error(f"[run_inference] Inference failed: {error_msg}")
        logger.info(f"[run_inference] Num predictions: {len(result['predictions'])}")
        if result['predictions']:
            top_conf = result['predictions'][0]['confidence']
            logger.info(f"[run_inference] Top confidence: {top_conf} (type: {type(top_conf).__name__}, is_nan: {top_conf != top_conf})")

        return result
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise

def _db_session() -> Session:
    return SessionLocal()

def _mark_running(db: Session, job_id: str):
    now = datetime.now(timezone.utc)
    job = db.get(Job, job_id)
    if not job:
        return
    
    # job is now running
    job.status = "running"
    job.started_at = now

    db.add(job)
    db.commit()

def _mark_done(db: Session, job_id: str, result: dict):
    now = datetime.now(timezone.utc)
    job = db.get(Job, job_id)
    if not job:
        return

    # Log before serialization
    logger.info(f"[_mark_done] Before serialization:")
    if result.get('predictions'):
        top_conf = result['predictions'][0]['confidence']
        logger.info(f"  Top confidence: {top_conf} (type: {type(top_conf).__name__})")

    # Ensure result is JSON-serializable (convert through JSON round-trip)
    # This prevents issues with numpy/torch types being stored as NaN
    try:
        result_json = json.loads(json.dumps(result))
        logger.info(f"[_mark_done] After JSON serialization:")
        if result_json.get('predictions'):
            top_conf = result_json['predictions'][0]['confidence']
            logger.info(f"  Top confidence: {top_conf} (type: {type(top_conf).__name__}, is_nan: {top_conf != top_conf})")
    except Exception as e:
        logger.error(f"[_mark_done] JSON serialization failed: {e}")
        result_json = result

    # job is now done
    job.status = "completed"
    job.finished_at = now
    job.result_json = result_json
    db.add(job)

    # insert a JobResult
    jr = JobResult(job_id=job.id, result=result_json)

    logger.info(f"[_mark_done] Storing to database, first confidence: {result_json['predictions'][0]['confidence'] if result_json.get('predictions') else 'N/A'}")
    db.add(jr)
    db.commit()

def _mark_error(db: Session, job_id: str, error_msg: str):
    now = datetime.now(timezone.utc)
    job = db.get(Job, job_id)
    if not job:
        return
    
    # job is now errored
    job.status = "error"
    job.finished_at = now
    job.result_json = {"error": error_msg}

    db.add(job)
    db.commit()

async def main():
    r = from_url(settings.redis_url, decode_responses=True)
    print(f"[worker] Connected to {settings.redis_url}; listening on '{settings.redis_queue}'")

    while True:
        job_id = None
        try:
            item = await r.blpop(settings.redis_queue, timeout=0)  # block forever
            if not item:
                continue

            # got a job
            _, payload = item
            job_msg = json.loads(payload)
            job_id = job_msg.get("job_id")
            sample_index = job_msg.get("sample_index", 0)
            # Use converted filepath for inference, fallback to raw filepath for compatibility
            filepath = job_msg.get("converted_filepath") or job_msg.get("filepath")

            # update DB -> running
            db = _db_session()
            try:
                _mark_running(db, job_id)
            finally:
                db.close()

            # Run model on specific sample
            result = run_inference(filepath, sample_index=sample_index)

            # update DB -> done
            db = _db_session()
            try:
                _mark_done(db, job_id, result)
            finally:
                db.close()

            # Publish completion status
            await r.publish('job_status_updates', json.dumps({
                'job_id': job_id,
                'status': 'completed',
                'result': result,
            }))
            
            logger.info(f"[worker] completed job {job_id}")

            print(f"[worker] completed job {job_id}")

        except Exception as e:
            if job_id:
                db = _db_session()
                try:
                    _mark_error(db, job_id, str(e))
                    
                    # Publish error status
                    await r.publish('job_status_updates', json.dumps({
                        'job_id': job_id,
                        'status': 'error',
                        'error': str(e),
                    }))
                finally:
                    db.close()
            
            logger.error(f"[worker] error: {e}")
            await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(main())