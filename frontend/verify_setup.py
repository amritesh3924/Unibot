"""Verify that the setup is correct before starting servers"""
import sys
import os
from pathlib import Path

def check_python_version():
    """Check if Python version is 3.8+"""
    if sys.version_info < (3, 8):
        print("[ERROR] Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"[OK] Python version: {sys.version.split()[0]}")
    return True

def check_dependencies():
    """Check if required packages are installed"""
    # Map package names to their import names
    required = {
        'fastapi': 'fastapi',
        'uvicorn': 'uvicorn',
        'langchain-groq': 'langchain_groq',
        'langgraph': 'langgraph',
        'langchain-chroma': 'langchain_chroma',
        'playwright': 'playwright',
        'chromadb': 'chromadb'
    }
    missing = []
    
    for package, import_name in required.items():
        try:
            __import__(import_name)
            print(f"[OK] {package} is installed")
        except ImportError:
            print(f"[ERROR] {package} is NOT installed")
            missing.append(package)
    
    if missing:
        print(f"\n[ERROR] Missing packages: {', '.join(missing)}")
        print("Install them with: pip install -r requirements.txt")
        return False
    return True

def check_env_file():
    """Check if .env file exists and has GROQ_API_KEY"""
    env_path = Path(__file__).parent.parent / '.env'
    
    if not env_path.exists():
        print(f"[WARNING] .env file not found at {env_path}")
        print("Create a .env file with GROQ_API_KEY=your_key")
        return False
    
    # Try to load and check
    try:
        from dotenv import load_dotenv
        load_dotenv(env_path, override=True)
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            print("[ERROR] GROQ_API_KEY not found in .env file")
            return False
        print("[OK] GROQ_API_KEY is set in .env")
        return True
    except Exception as e:
        print(f"[ERROR] Could not read .env file: {e}")
        return False

def main():
    print("=" * 50)
    print("UniBot Setup Verification")
    print("=" * 50)
    print()
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Environment File", check_env_file),
    ]
    
    all_ok = True
    for name, check_func in checks:
        print(f"\nChecking {name}...")
        if not check_func():
            all_ok = False
    
    print()
    print("=" * 50)
    if all_ok:
        print("[OK] All checks passed! You can start the servers.")
        return 0
    else:
        print("[ERROR] Some checks failed. Please fix the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())