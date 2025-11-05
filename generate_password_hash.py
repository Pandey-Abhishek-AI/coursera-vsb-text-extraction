#!/usr/bin/env python3
"""
Password Hash Generator for Streamlit Authenticator
Usage: python generate_password_hash.py
"""

import streamlit_authenticator as stauth
import sys

def generate_password_hash(password: str) -> str:
    """Generate a bcrypt hash for the given password."""
    return stauth.Hasher([password]).generate()[0]

def main():
    """Interactive password hash generator."""
    print("🔐 VSB Password Hash Generator")
    print("=" * 40)
    print("Generate secure password hashes for user management")
    print()
    
    while True:
        try:
            # Get password input
            password = input("Enter password (or 'quit' to exit): ")
            
            if password.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if len(password) < 6:
                print("⚠️  Password should be at least 6 characters long")
                continue
            
            # Generate hash
            hashed_password = generate_password_hash(password)
            
            print(f"✅ Generated hash: {hashed_password}")
            print()
            print("📝 Add this to your secrets.toml file:")
            print(f'password = "{hashed_password}"  # password: {password}')
            print()
            print("-" * 40)
            print()
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 