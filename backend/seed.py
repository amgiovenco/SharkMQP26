#!/usr/bin/env python
"""
seed.py: Seed initial admin user and default organization into the database.
"""

import secrets
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from app.db import engine, Base
from app.models import User, UserRole, Organization, OrganizationMembership, OrganizationRole, RegistrationCode
from app.auth import hash_password
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def seed_database():
    # Create all tables
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created")

    db = Session(bind=engine)

    try:
        # 1. Create default organization
        org_slug = "default"
        existing_org = db.query(Organization).filter(Organization.slug == org_slug).first()

        if not existing_org:
            org = Organization(
                name="Default Organization",
                slug=org_slug,
                description="Default organization for initial setup",
                status="active",
                created_at=datetime.now(timezone.utc)
            )
            db.add(org)
            db.flush()
            logger.info(f"Created organization: {org.name} (id={org.id})")
        else:
            org = existing_org
            logger.info(f"ℹOrganization already exists: {org.name} (id={org.id})")

        # 2. Create admin user
        email = "cejason@wpi.edu"
        password = "wpi"
        role = UserRole.admin

        existing_user = db.query(User).filter(User.email == email).first()

        if not existing_user:
            user = User(
                email=email,
                password_hash=hash_password(password),
                role=role,
                first_name="Connor",
                last_name="Jason",
                job_title="Administrator",
                is_system_admin=True,  # Make system admin
            )
            db.add(user)
            db.flush()
            logger.info(f"Created admin user: {email} (is_system_admin=True)")
        else:
            user = existing_user
            # Update to be system admin
            user.is_system_admin = True
            logger.info(f"User already exists: {email}")

        # 3. Add user to organization as owner
        existing_membership = db.query(OrganizationMembership).filter(
            OrganizationMembership.organization_id == org.id,
            OrganizationMembership.user_id == user.id
        ).first()

        if not existing_membership:
            membership = OrganizationMembership(
                organization_id=org.id,
                user_id=user.id,
                role=OrganizationRole.owner,
                status="active",
                joined_at=datetime.now(timezone.utc)
            )
            db.add(membership)
            logger.info(f"Added {email} to {org.name} as owner")
        else:
            logger.info(f"ℹ{email} already member of {org.name}")

        # 4. Create sample registration codes with random codes
        sample_roles = [
            (OrganizationRole.admin, "Admin registration code"),
            (OrganizationRole.researcher, "Researcher registration code"),
            (OrganizationRole.member, "Member registration code"),
        ]

        for code_role, description in sample_roles:
            # Generate random code: SHARK-XXXXXX (6 random hex chars)
            code_str = f"SHARK-{secrets.token_hex(3).upper()}"

            # Ensure uniqueness (unlikely but possible)
            while db.query(RegistrationCode).filter(
                RegistrationCode.code == code_str
            ).first():
                code_str = f"SHARK-{secrets.token_hex(3).upper()}"

            reg_code = RegistrationCode(
                organization_id=org.id,
                code=code_str,
                role=code_role,
                created_by=user.id,
                created_at=datetime.now(timezone.utc),
                expires_at=None,
                uses_remaining=None,  # Unlimited
                times_used=0,
                status="active"
            )
            db.add(reg_code)
            logger.info(f"Created registration code: {code_str} ({code_role.value})")

        db.commit()

        # Fetch all registration codes to display
        all_codes = db.query(RegistrationCode).filter(
            RegistrationCode.organization_id == org.id
        ).all()

        logger.info("\n" + "="*60)
        logger.info("Database seeding completed successfully!")
        logger.info("="*60)
        logger.info(f"\nLogin credentials:")
        logger.info(f"  Username: cejason")
        logger.info(f"  Password: wpi")
        logger.info(f"\nRegistration codes (unlimited use):")
        for code in all_codes:
            logger.info(f"  {code.code} ({code.role.value})")

    except Exception as e:
        db.rollback()
        logger.error(f"Seeding failed: {e}", exc_info=True)
        raise
    finally:
        db.close()

if __name__ == "__main__":
    seed_database()
