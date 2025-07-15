import os
import requests
from dotenv import load_dotenv

load_dotenv()

PAYSTACK_SECRET_KEY = os.getenv('PAYSTACK_SECRET_KEY')
PAYSTACK_PUBLIC_KEY = os.getenv('PAYSTACK_PUBLIC_KEY')
BASE_URL = "https://api.paystack.co"

def verify_payment(reference):
    """Verify a Paystack payment"""
    url = f"{BASE_URL}/transaction/verify/{reference}"
    headers = {
        "Authorization": f"Bearer {PAYSTACK_SECRET_KEY}",
        "Content-Type": "application/json",
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    return None

def initialize_payment(email, amount, plan_id=None):
    """Initialize Paystack payment"""
    url = f"{BASE_URL}/transaction/initialize"
    headers = {
        "Authorization": f"Bearer {PAYSTACK_SECRET_KEY}",
        "Content-Type": "application/json",
    }
    data = {
        "email": email,
        "amount": int(amount) * 100,  # Convert to kobo
        "currency": "NGN",  # or your preferred currency
    }
    if plan_id:
        data["plan"] = plan_id
    
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        return response.json()
    return None