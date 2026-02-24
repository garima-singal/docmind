import openai
import chromadb
import langchain
import streamlit
from dotenv import load_dotenv
import os

load_dotenv()

print("✅ All imports successful")
print(f"✅ OpenAI Key loaded: {'Yes' if os.getenv('OPENAI_API_KEY') else 'NO - check .env'}")
print(f"✅ LangChain version: {langchain.__version__}")
print(f"✅ ChromaDB version: {chromadb.__version__}")