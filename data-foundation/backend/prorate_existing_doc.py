#!/usr/bin/env python3

import requests

# Use the document that's visible in the web app
doc_id = "unknown_document_20250826_162829_388"
url = f"http://localhost:8000/api/v1/prorating/process/{doc_id}"

payload = {
    "document_id": doc_id,
    "facility_info": [
        {
            "facility_id": "facility_001",
            "name": "Main Facility",
            "percentage": 60.0
        },
        {
            "facility_id": "facility_002", 
            "name": "Secondary Facility",
            "percentage": 40.0
        }
    ],
    "method": "custom"
}

print(f"Creating prorated allocations for document: {doc_id}")
print(f"Calling: {url}")

try:
    response = requests.post(url, json=payload)
    print(f"\nResponse status: {response.status_code}")
    if response.status_code == 200:
        print("Success! MonthlyUsageAllocation nodes created.")
        print("\nNow refresh the frontend and check if the prorated value is displayed.")
    else:
        print(f"Error: {response.json()}")
except Exception as e:
    print(f"Error: {e}")