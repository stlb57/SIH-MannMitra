# check_env.py

import os
from dotenv import load_dotenv

load_dotenv()

key = os.getenv("KEY")
skey = os.getenv("skey")
secret_key = os.getenv("SECRET_KEY") # Check for the Flask secret key

print("\n--- Final Environment Variable Check ---")
print(f"Chat Agent KEY:   -> {key}")
print(f"Search Agent skey:  -> {skey}")
print(f"Flask SECRET_KEY:   -> {secret_key}") # Print the result
print("--------------------------------------\n")

if key and skey and secret_key:
    print("✅ SUCCESS: All keys are loaded correctly!")
else:
    print("❌ FAILED: One or more keys are missing from your .env file.")
    if not secret_key:
        print("   >>> The 'SECRET_KEY' variable is missing. This is causing the login error.")