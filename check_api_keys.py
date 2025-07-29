#!/usr/bin/env python3
"""
Fast API Key Validator for Gemini Keys
Checks all GEMINI_API_KEY_1 through GEMINI_API_KEY_46 
"""

import os
import requests
import concurrent.futures
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_api_key(key_index):
    """Test a single API key with simple embedding call - DETAILED VERSION"""
    key_name = f"GEMINI_API_KEY_{key_index}"
    api_key = os.getenv(key_name)
    
    if not api_key or api_key.startswith("YOUR_API_KEY") or len(api_key.strip()) < 10:
        return {
            'key_index': key_index,
            'key_name': key_name,
            'status': 'MISSING',
            'error': 'Environment variable not set or placeholder value'
        }
    
    # Simple embedding test with "hi"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent?key={api_key}"
    
    payload = {
        "model": "models/gemini-embedding-001",
        "content": {
            "parts": [{"text": "hi"}]
        }
    }
    
    headers = {"Content-Type": "application/json"}
    
    # SHOW DETAILED NETWORK ACTIVITY
    print(f"🌐 Testing {key_name}: Making network call...")
    
    try:
        import time
        start_time = time.time()
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        end_time = time.time()
        
        print(f"   ⏱️  Response: {response.status_code} in {end_time - start_time:.2f}s")
        
        if response.status_code == 200:
            # Verify we got actual embedding data
            data = response.json()
            if 'embedding' in data and 'values' in data['embedding']:
                values_count = len(data['embedding']['values'])
                print(f"   ✅ SUCCESS: Got {values_count} embedding values")
            else:
                print(f"   ⚠️  Got 200 but no embedding data: {data}")
            
            return {
                'key_index': key_index,
                'key_name': key_name,
                'status': 'VALID',
                'error': None
            }
        elif response.status_code == 403:
            print(f"   ❌ INVALID: 403 Forbidden")
            return {
                'key_index': key_index,
                'key_name': key_name,
                'status': 'INVALID',
                'error': 'API key invalid or permissions denied'
            }
        elif response.status_code == 429:
            print(f"   🔄 RATE LIMITED: 429 Too Many Requests")
            return {
                'key_index': key_index,
                'key_name': key_name,
                'status': 'RATE_LIMITED',
                'error': 'API key valid but rate limited'
            }
        else:
            print(f"   🔥 ERROR: {response.status_code} - {response.text[:100]}")
            return {
                'key_index': key_index,
                'key_name': key_name,
                'status': 'ERROR',
                'error': f'HTTP {response.status_code}: {response.text[:100]}'
            }
            
    except requests.exceptions.RequestException as e:
        print(f"   🔥 CONNECTION ERROR: {str(e)}")
        return {
            'key_index': key_index,
            'key_name': key_name,
            'status': 'CONNECTION_ERROR',
            'error': str(e)
        }

def main():
    print("🔑 Testing Gemini API Keys (1-46)...")
    print("=" * 60)
    
    # Test all keys in parallel for speed
    with concurrent.futures.ThreadPoolExecutor(max_workers=46) as executor:
        futures = [executor.submit(test_api_key, i) for i in range(1, 47)]
        results = [future.result() for future in futures]
    
    # Sort results by key index
    results.sort(key=lambda x: x['key_index'])
    
    # Print results
    valid_count = 0
    invalid_count = 0
    missing_count = 0
    rate_limited_count = 0
    
    for result in results:
        status = result['status']
        key_name = result['key_name']
        
        if status == 'VALID':
            print(f"✅ {key_name}: VALID")
            valid_count += 1
        elif status == 'INVALID':
            print(f"❌ {key_name}: INVALID - {result['error']}")
            invalid_count += 1
        elif status == 'MISSING':
            print(f"⚠️  {key_name}: MISSING - {result['error']}")
            missing_count += 1
        elif status == 'RATE_LIMITED':
            print(f"🔄 {key_name}: RATE LIMITED (but valid)")
            valid_count += 1  # Count as valid since key works
            rate_limited_count += 1
        else:
            print(f"🔥 {key_name}: ERROR - {result['error']}")
            invalid_count += 1
    
    print("=" * 60)
    print(f"📊 SUMMARY:")
    print(f"✅ Valid keys: {valid_count}")
    print(f"❌ Invalid keys: {invalid_count}")
    print(f"⚠️  Missing keys: {missing_count}")
    if rate_limited_count > 0:
        print(f"🔄 Rate limited keys: {rate_limited_count}")
    
    print(f"\n🎯 TOTAL WORKING KEYS: {valid_count}/46")
    
    if invalid_count > 0 or missing_count > 0:
        print(f"\n🚨 ACTION NEEDED:")
        print(f"   - Fix {invalid_count + missing_count} problematic keys")
        print(f"   - Check your .env file")

if __name__ == "__main__":
    main()
