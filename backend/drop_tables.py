from sqlalchemy import create_engine, MetaData

DATABASE_URL = "postgresql+psycopg2://admin:wpi_admin@localhost:5432/sharkid"
engine = create_engine(DATABASE_URL)

meta = MetaData()
meta.reflect(bind=engine)
meta.drop_all(bind=engine)

print("✅ All tables dropped successfully.")
