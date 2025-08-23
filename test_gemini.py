#!/usr/bin/env python
"""
Test Gemini API directly to debug the issue
"""
import os
import sys
import json

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from logic.text_generation.restore_missing_text import gemini_rest_generate
from config import settings

def test_gemini_basic():
    """Test basic Gemini API functionality"""
    print("🧪 Testing Gemini API...")
    print(f"API Key: {settings.GEMINI_API_KEY[:10]}..." if settings.GEMINI_API_KEY else "❌ No API Key")
    print("=" * 60)
    
    # Test 1: Simple prompt
    print("🧪 Test 1: Simple prompt")
    simple_prompt = "Hello, how are you? Please respond with just 'I am fine.'"
    
    try:
        response = gemini_rest_generate(
            prompt=simple_prompt,
            api_key=settings.GEMINI_API_KEY,
            temperature=0.1,
            max_output_tokens=50
        )
        print(f"✅ Response: {response}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("-" * 60)
    
    # Test 2: Text completion like our use case
    print("🧪 Test 2: Text completion (like our use case)")
    completion_prompt = """
You are helping restore missing audio in a transcript.

Context before: "Thank you. Thank you."
Context after: "In 25 years, global energy consumption will increase by 50%."

What word or phrase was likely said in between? Return ONLY the missing text.
"""
    
    try:
        response = gemini_rest_generate(
            prompt=completion_prompt,
            api_key=settings.GEMINI_API_KEY,
            temperature=0.3,
            max_output_tokens=100
        )
        print(f"✅ Response: {response}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("-" * 60)
    
    # Test 3: Test with different model
    print("🧪 Test 3: Test with gemini-1.5-pro model")
    try:
        response = gemini_rest_generate(
            prompt="Complete: 'The sky is'",
            api_key=settings.GEMINI_API_KEY,
            model="gemini-1.5-pro",
            temperature=0.1,
            max_output_tokens=20
        )
        print(f"✅ Response: {response}")
    except Exception as e:
        print(f"❌ Error: {e}")

def test_api_key_validity():
    """Test if API key is valid by making a direct request"""
    import requests
    
    print("\n🔑 Testing API Key validity...")
    
    api_key = settings.GEMINI_API_KEY
    if not api_key:
        print("❌ No API key found")
        return
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    
    body = {
        "contents": [
            {
                "role": "user", 
                "parts": [{"text": "Hello"}]
            }
        ]
    }
    
    try:
        response = requests.post(url, json=body, timeout=10)
        print(f"📡 Status Code: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ API Key is valid!")
            data = response.json()
            print(f"📝 Response preview: {str(data)[:200]}...")
        else:
            print(f"❌ API Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Network Error: {e}")

def test_network_connectivity():
    """Test basic network connectivity to Google APIs"""
    import requests
    
    print("\n🌐 Testing network connectivity...")
    
    try:
        # Test basic Google connectivity
        response = requests.get("https://www.google.com", timeout=5)
        print(f"✅ Google.com: {response.status_code}")
        
        # Test Gemini API endpoint
        response = requests.get("https://generativelanguage.googleapis.com", timeout=5)
        print(f"✅ Gemini API endpoint: {response.status_code}")
        
    except Exception as e:
        print(f"❌ Network connectivity issue: {e}")

if __name__ == "__main__":
    print("🔍 GEMINI API DEBUGGING")
    print("=" * 60)
    
    test_network_connectivity()
    test_api_key_validity() 
    test_gemini_basic()
    
    print("\n" + "=" * 60)
    print("🔍 Debug complete!")
