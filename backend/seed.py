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
        password = "wpiwpiwpi"
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

        # 3b. Create second admin user
        email2 = "amgiovenco@wpi.edu"
        password2 = "wpiwpiwpi"
        role2 = UserRole.admin

        existing_user2 = db.query(User).filter(User.email == email2).first()

        if not existing_user2:
            user2 = User(
                email=email2,
                password_hash=hash_password(password2),
                role=role2,
                first_name="Ally",
                last_name="Giovenco",
                job_title="Administrator",
                is_system_admin=True,
            )
            db.add(user2)
            db.flush()
            logger.info(f"Created admin user: {email2} (is_system_admin=True)")
        else:
            user2 = existing_user2
            user2.is_system_admin = True
            logger.info(f"User already exists: {email2}")

        # Add second user to organization as owner
        existing_membership2 = db.query(OrganizationMembership).filter(
            OrganizationMembership.organization_id == org.id,
            OrganizationMembership.user_id == user2.id
        ).first()

        if not existing_membership2:
            membership2 = OrganizationMembership(
                organization_id=org.id,
                user_id=user2.id,
                role=OrganizationRole.owner,
                status="active",
                joined_at=datetime.now(timezone.utc)
            )
            db.add(membership2)
            logger.info(f"Added {email2} to {org.name} as owner")
        else:
            logger.info(f"ℹ{email2} already member of {org.name}")

        # 3c. Create fourth admin user
        email_cz = "czhang12@wpi.edu"
        password_cz = "wpiwpiwpi"

        existing_user_cz = db.query(User).filter(User.email == email_cz).first()

        if not existing_user_cz:
            user_cz = User(
                email=email_cz,
                password_hash=hash_password(password_cz),
                role=UserRole.admin,
                first_name="Chengqian",
                last_name="Zhang",
                job_title="Administrator",
                is_system_admin=True,
            )
            db.add(user_cz)
            db.flush()
            logger.info(f"Created admin user: {email_cz} (is_system_admin=True)")
        else:
            user_cz = existing_user_cz
            user_cz.is_system_admin = True
            logger.info(f"User already exists: {email_cz}")

        # Add czhang12 to organization as owner
        existing_membership_cz = db.query(OrganizationMembership).filter(
            OrganizationMembership.organization_id == org.id,
            OrganizationMembership.user_id == user_cz.id
        ).first()

        if not existing_membership_cz:
            membership_cz = OrganizationMembership(
                organization_id=org.id,
                user_id=user_cz.id,
                role=OrganizationRole.owner,
                status="active",
                joined_at=datetime.now(timezone.utc)
            )
            db.add(membership_cz)
            logger.info(f"Added {email_cz} to {org.name} as owner")
        else:
            logger.info(f"ℹ{email_cz} already member of {org.name}")

        # 3d. Create third admin user
        email3 = "kmlee@wpi.edu"
        password3 = "wpiwpiwpi"
        role3 = UserRole.admin

        existing_user3 = db.query(User).filter(User.email == email3).first()

        if not existing_user3:
            user3 = User(
                email=email3,
                password_hash=hash_password(password3),
                role=role3,
                first_name="Kyumin",
                last_name="Lee",
                job_title="Administrator",
                is_system_admin=True,
            )
            db.add(user3)
            db.flush()
            logger.info(f"Created admin user: {email3} (is_system_admin=True)")
        else:
            user3 = existing_user3
            user3.is_system_admin = True
            logger.info(f"User already exists: {email3}")

        # Add third user to organization as owner
        existing_membership3 = db.query(OrganizationMembership).filter(
            OrganizationMembership.organization_id == org.id,
            OrganizationMembership.user_id == user3.id
        ).first()

        if not existing_membership3:
            membership3 = OrganizationMembership(
                organization_id=org.id,
                user_id=user3.id,
                role=OrganizationRole.owner,
                status="active",
                joined_at=datetime.now(timezone.utc)
            )
            db.add(membership3)
            logger.info(f"Added {email3} to {org.name} as owner")
        else:
            logger.info(f"ℹ{email3} already member of {org.name}")

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
        logger.info(f"\n  Username: amgiovenco")
        logger.info(f"  Password: wpiwpiwpi")
        logger.info(f"\n  Username: kmlee")
        logger.info(f"  Password: wpiwpiwpi")
        logger.info(f"\n  Username: czhang12")
        logger.info(f"  Password: wpiwpiwpi")
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
