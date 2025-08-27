#!/usr/bin/env python3

import requests

doc_id = "2853f82f-3f49-4b8c-a763-600a0fb8ce1e"
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

print(f"Calling prorating API: {url}")
print(f"Payload: {payload}")

try:
    response = requests.post(url, json=payload)
    print(f"\nResponse status: {response.status_code}")
    print(f"Response body: {response.json()}")
    
    if response.status_code == 201:
        print("\nSuccess! MonthlyUsageAllocation nodes should now be created.")
        print("You can verify in the frontend by viewing the electricity bill details.")
except Exception as e:
    print(f"Error calling API: {e}")