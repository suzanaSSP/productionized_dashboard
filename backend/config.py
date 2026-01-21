from dotenv import load_dotenv
import os

load_dotenv()

class Config:
    DEBUG = os.getenv('DEBUG', 'False') == 'True'
    TESTING = os.getenv('TESTING', 'False') == 'True'
    # Fixed to use PostgreSQL:
    DATABASE_URI = os.getenv('DATABASE_URI', 'postgresql://byu_student:your_secure_password@localhost:5432/olist_sales')
    SECRET_KEY = os.getenv('SECRET_KEY', 'your_secret_key_here')