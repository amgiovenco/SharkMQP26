#!/usr/bin/env python3
"""
Script to clear all jobs and cases from the database.
Run from the backend directory: python clear_data.py
"""

from app.db import SessionLocal
from app.models import Case, Job, JobResult

def clear_all_data():
    db = SessionLocal()
    try:
        # Delete all job results first (they cascade from jobs, but being explicit)
        job_result_count = db.query(JobResult).delete(synchronize_session=False)
        print(f"Deleted {job_result_count} job results")

        # Delete all jobs
        job_count = db.query(Job).delete(synchronize_session=False)
        print(f"Deleted {job_count} jobs")

        # Delete all cases
        case_count = db.query(Case).delete(synchronize_session=False)
        print(f"Deleted {case_count} cases")

        db.commit()
        print("\nAll data cleared successfully!")

    except Exception as e:
        db.rollback()
        print(f"Error clearing data: {e}")
        raise
    finally:
        db.close()

if __name__ == "__main__":
    clear_all_data()
