"""
F1 2021 AI Race Engineer - Root Launcher
Redirects execution into ai_engineer/main.py
"""
import os
import sys

# Redirect to ai_engineer directory
ai_engineer_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ai_engineer")
sys.path.insert(0, ai_engineer_dir)

from main import main

if __name__ == "__main__":
    main()
