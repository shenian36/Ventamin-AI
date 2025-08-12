#!/usr/bin/env python3
"""
Git Configuration Helper for Ventamin AI
This script helps you configure Git with your GitHub information
"""

import subprocess
import sys
import os

def run_command(command):
    """Run a command and return the result"""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def check_git_installed():
    """Check if Git is installed"""
    success, stdout, stderr = run_command("git --version")
    if success:
        print(f"✅ Git is installed: {stdout.strip()}")
        return True
    else:
        print("❌ Git is not installed or not in PATH")
        print("Please install Git from: https://git-scm.com/download/win")
        return False

def configure_git():
    """Configure Git with user information"""
    print("\n🔧 Configuring Git...")
    
    # Get user input
    print("Please enter your Git configuration details:")
    name = input("Full Name (e.g., John Doe): ").strip()
    email = input("Email Address: ").strip()
    
    if not name or not email:
        print("❌ Name and email are required!")
        return False
    
    # Configure Git
    print(f"\nSetting Git configuration...")
    
    # Set user name
    success, stdout, stderr = run_command(f'git config --global user.name "{name}"')
    if success:
        print(f"✅ Git user name set to: {name}")
    else:
        print(f"❌ Failed to set user name: {stderr}")
        return False
    
    # Set user email
    success, stdout, stderr = run_command(f'git config --global user.email "{email}"')
    if success:
        print(f"✅ Git user email set to: {email}")
    else:
        print(f"❌ Failed to set user email: {stderr}")
        return False
    
    # Set default branch to main
    success, stdout, stderr = run_command("git config --global init.defaultBranch main")
    if success:
        print("✅ Default branch set to 'main'")
    else:
        print(f"⚠️ Warning: Could not set default branch: {stderr}")
    
    # Set line ending configuration for Windows
    success, stdout, stderr = run_command("git config --global core.autocrlf true")
    if success:
        print("✅ Line ending configuration set for Windows")
    else:
        print(f"⚠️ Warning: Could not set line ending config: {stderr}")
    
    return True

def show_git_config():
    """Show current Git configuration"""
    print("\n📋 Current Git Configuration:")
    
    config_items = [
        ("user.name", "User Name"),
        ("user.email", "User Email"),
        ("init.defaultBranch", "Default Branch"),
        ("core.autocrlf", "Line Ending Config")
    ]
    
    for config_key, description in config_items:
        success, stdout, stderr = run_command(f"git config --global {config_key}")
        if success:
            value = stdout.strip()
            print(f"  {description}: {value}")
        else:
            print(f"  {description}: Not set")

def initialize_repository():
    """Initialize Git repository for Ventamin AI"""
    print("\n🚀 Initializing Git Repository...")
    
    # Check if already initialized
    if os.path.exists(".git"):
        print("✅ Git repository already initialized")
        return True
    
    # Initialize repository
    success, stdout, stderr = run_command("git init")
    if success:
        print("✅ Git repository initialized")
        return True
    else:
        print(f"❌ Failed to initialize repository: {stderr}")
        return False

def main():
    """Main configuration function"""
    print("=" * 50)
    print("🔧 Git Configuration Helper for Ventamin AI")
    print("=" * 50)
    
    # Check if Git is installed
    if not check_git_installed():
        print("\n📥 Please install Git first, then run this script again.")
        input("Press Enter to exit...")
        return
    
    # Configure Git
    if configure_git():
        print("\n✅ Git configuration completed successfully!")
        
        # Show configuration
        show_git_config()
        
        # Initialize repository
        if initialize_repository():
            print("\n🎉 Your Ventamin AI project is now ready for GitHub!")
            print("\n📋 Next steps:")
            print("1. Create a repository on GitHub")
            print("2. Add your files: git add .")
            print("3. Commit: git commit -m 'Initial commit'")
            print("4. Add remote: git remote add origin <your-github-url>")
            print("5. Push: git push -u origin main")
        else:
            print("\n❌ Failed to initialize repository")
    else:
        print("\n❌ Git configuration failed")
    
    print("\n" + "=" * 50)
    input("Press Enter to exit...")

if __name__ == "__main__":
    main()
