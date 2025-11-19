from sqlalchemy import create_engine, Column, Integer, String, Date, Time, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
# Replace with your MySQL details
# DATABASE_URL = "mysql+pymysql://root:'@123'@localhost/hba"
from sqlalchemy import create_engine
import os
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL") 
engine = create_engine(DATABASE_URL)
try:
    with engine.connect() as connection:
        print("✅ Database connection successful!")
except Exception as e:
    print("❌ Database connection failed!")
    print(f"Error: {e}")

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


