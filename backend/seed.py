#!/usr/bin/env python
"""
seed.py: Seed initial admin user into the database.
"""

from sqlalchemy.orm import Session
from app.db import engine
from app.models import User, UserRole
from app.auth import hash_password
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def seed_admin():
    db = Session(bind=engine)

    username = "cejason"
    password = "wpi"
    role = UserRole.admin

    existing = db.query(User).filter(User.username == username).first()
    if existing:
        logger.info(f"User '{username}' already exists. Skipping.")
        return

    user = User(
        username=username,
        password_hash=hash_password(password),
        role=role,
        first_name="Connor",
        last_name="Jason",
        job_title="Administrator"
    )

    db.add(user)
    db.commit()
    logger.info(f"✅ Created admin user: {username} (role={role.value})")

if __name__ == "__main__":
    seed_admin()
