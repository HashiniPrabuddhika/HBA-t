import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration settings
DEEPSEEK_API_KEY = os.getenv("OPENAI_API_KEY2")