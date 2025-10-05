import json
import time
from datetime import datetime, timezone

from redis.asyncio import from_url
from sqlalchemy.orm import Session

from app.db import SessionLocal
from app.models import Job, JobResult
from app.settings import settings  # fixed relative import

def run_inference(filepath: str):
    # TODO: real ML inference
    return {
        "winner": "Carcharhinus leucas",
        "topk": [
            {"label": "Carcharhinus leucas", "prob": 0.82},
            {"label": "Carcharhinus limbatus", "prob": 0.12},
            {"label": "Sphyrna lewini", "prob": 0.06},
        ],
        "source_file": filepath,
    }

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

    # job is now done
    job.status = "done"
    job.finished_at = now
    job.result_json = result
    db.add(job)

    # insert a JobResult
    jr = JobResult(job_id=job.id, result=result)

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

            _qname, payload = item
            job_msg = json.loads(payload)
            job_id = job_msg.get("job_id")
            filepath = job_msg.get("filepath")

            # update DB -> running
            db = _db_session()
            try:
                _mark_running(db, job_id)
            finally:
                db.close()

            # Run model
            result = run_inference(filepath)

            # update DB -> done
            db = _db_session()
            try:
                _mark_done(db, job_id, result)
            finally:
                db.close()

            print(f"[worker] completed job {job_id}")

        except Exception as e:
            # if job_id known, mark error
            try:
                if job_id:
                    db = _db_session()
                    try:
                        _mark_error(db, job_id, str(e))
                    finally:
                        db.close()
            except Exception:
                pass
            print(f"[worker] error: {e}")
            time.sleep(1)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
