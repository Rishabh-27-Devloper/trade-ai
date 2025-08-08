#!/usr/bin/env python3
"""
Verification script for Trading AI System
Checks all components and dependencies
"""
import os
import sys
import importlib.util
import sqlite3
import requests
from pathlib import Path

def check_file_exists(filepath, description):
    """Check if a file exists"""
    if Path(filepath).exists():
        print(f"✅ {description}: {filepath}")
        return True
    else:
        print(f"❌ Missing {description}: {filepath}")
        return False

def check_python_import(module_name, description):
    """Check if Python module can be imported"""
    try:
        importlib.import_module(module_name)
        print(f"✅ {description} module available")
        return True
    except ImportError as e:
        print(f"❌ {description} module missing: {e}")
        return False

def check_directory_structure():
    """Check if all required directories exist"""
    print("\n📁 Checking Directory Structure...")

    required_dirs = [
        "app",
        "app/models", 
        "app/ui/templates",
        "deploy",
        "tests"
    ]

    all_good = True
    for dir_path in required_dirs:
        if Path(dir_path).is_dir():
            print(f"✅ Directory exists: {dir_path}")
        else:
            print(f"❌ Missing directory: {dir_path}")
            all_good = False

    return all_good

def check_core_files():
    """Check if all core files exist"""
    print("\n📄 Checking Core Files...")

    required_files = [
        ("main.py", "Main application entry point"),
        ("requirements.txt", "Python dependencies"),
        ("Dockerfile", "Docker container definition"),
        ("README.md", "Documentation")
    ]

    all_good = True
    for filepath, description in required_files:
        if not check_file_exists(filepath, description):
            all_good = False

    return all_good

def main():
    """Main verification function"""
    print("🔍 Trading AI System Verification")
    print("=" * 50)

    checks = [
        ("Directory Structure", check_directory_structure),
        ("Core Files", check_core_files)
    ]

    results = {}
    for check_name, check_func in checks:
        try:
            results[check_name] = check_func()
        except Exception as e:
            print(f"❌ {check_name} check failed with error: {e}")
            results[check_name] = False

    # Summary
    print("\n" + "=" * 50)
    print("📋 Verification Summary")
    print("=" * 50)

    passed = sum(results.values())
    total = len(results)

    for check_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{check_name:<20}: {status}")

    print(f"\nOverall: {passed}/{total} checks passed")

    if passed == total:
        print("\n🎉 System structure looks good!")
        print("\n🚀 Next steps:")
        print("   1. Copy .env.example to .env and configure")
        print("   2. Run: docker-compose up --build") 
        print("   3. Access dashboard at: http://localhost:5000")
        return True
    else:
        print("\n⚠️ Some checks failed. Please fix the issues.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)