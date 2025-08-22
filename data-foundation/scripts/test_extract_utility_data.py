#!/usr/bin/env python3
"""
Extract key utility bill data from the PDF using PyMuPDF.
"""

import os
import sys
import re
import json
from pathlib import Path
import fitz  # PyMuPDF

def extract_utility_bill_data(pdf_path):
    """Extract structured data from utility bill PDF."""
    
    # Open PDF
    pdf_document = fitz.open(pdf_path)
    
    # Initialize data structure
    extracted_data = {
        "account_number": None,
        "statement_date": None,
        "billing_period_start": None,
        "billing_period_end": None,
        "total_kwh": None,
        "peak_kwh": None,
        "off_peak_kwh": None,
        "total_amount_due": None,
        "due_date": None,
        "service_address": None,
        "meter_readings": [],
        "charges": []
    }
    
    # Extract text from all pages
    full_text = ""
    for page_num in range(pdf_document.page_count):
        page = pdf_document[page_num]
        full_text += page.get_text() + "\n"
    
    # Extract specific fields using regex patterns
    patterns = {
        "account_number": r"Account Number:\s*([^\n]+)",
        "statement_date": r"Statement Date:\s*([^\n]+)",
        "amount_due": r"Amount Due:\s*\$?([0-9,]+\.?\d*)",
        "due_date": r"Due Date:\s*([^\n]+)",
        "billing_period": r"Billing Period:\s*([^to]+)\s*to\s*([^\n]+)",
        "total_consumption": r"Total Consumption:\s*([0-9,]+)\s*kWh",
        "peak_usage": r"Peak Demand Usage\s*\(([0-9,]+)\s*kWh\)",
        "off_peak_usage": r"Oﬀ-Peak Demand Usage\s*\(([0-9,]+)\s*kWh\)",
        "service_for": r"SERVICE FOR:\s*([^\n]+)"
    }
    
    # Extract using patterns
    for key, pattern in patterns.items():
        match = re.search(pattern, full_text)
        if match:
            if key == "billing_period":
                extracted_data["billing_period_start"] = match.group(1).strip()
                extracted_data["billing_period_end"] = match.group(2).strip()
            elif key == "total_consumption":
                extracted_data["total_kwh"] = float(match.group(1).replace(",", ""))
            elif key == "peak_usage":
                extracted_data["peak_kwh"] = float(match.group(1).replace(",", ""))
            elif key == "off_peak_usage":
                extracted_data["off_peak_kwh"] = float(match.group(1).replace(",", ""))
            elif key == "amount_due":
                extracted_data["total_amount_due"] = float(match.group(1).replace(",", ""))
            elif key == "service_for":
                extracted_data["service_address"] = match.group(1).strip()
            else:
                extracted_data[key] = match.group(1).strip()
    
    # Extract meter readings
    meter_pattern = r"(MTR-\d+-[A-Z])\s+([^\\n]+)\s+([0-9,]+)\s*kWh\s+([0-9,]+)\s*kWh"
    meter_matches = re.findall(meter_pattern, full_text)
    for match in meter_matches:
        extracted_data["meter_readings"].append({
            "meter_id": match[0],
            "service_type": match[1].strip(),
            "previous_reading": float(match[2].replace(",", "")),
            "current_reading": float(match[3].replace(",", ""))
        })
    
    # Extract charges
    charge_pattern = r"([^\\n]+)\s+\$([0-9,]+\.?\d*)\s*/\s*kWh\s+\$([0-9,]+\.?\d*)"
    charge_matches = re.findall(charge_pattern, full_text)
    for match in charge_matches:
        extracted_data["charges"].append({
            "description": match[0].strip(),
            "rate": float(match[1].replace(",", "")),
            "amount": float(match[2].replace(",", ""))
        })
    
    pdf_document.close()
    
    return extracted_data

def main():
    file_path = "/Users/eugene/dev/ai/agentos/ehs-ai-demo/data-foundation/data/electric_bill.pdf"
    
    print(f"Extracting data from: {file_path}")
    print("-" * 50)
    
    try:
        # Extract data
        data = extract_utility_bill_data(file_path)
        
        # Display results
        print("\nExtracted Utility Bill Data:")
        print("=" * 50)
        print(f"Account Number: {data['account_number']}")
        print(f"Statement Date: {data['statement_date']}")
        print(f"Due Date: {data['due_date']}")
        print(f"Amount Due: ${data['total_amount_due']:,.2f}")
        print(f"\nBilling Period: {data['billing_period_start']} to {data['billing_period_end']}")
        print(f"Service Address: {data['service_address']}")
        print(f"\nEnergy Consumption:")
        print(f"  Total: {data['total_kwh']:,} kWh" if data['total_kwh'] else "  Total: Not found")
        print(f"  Peak: {data['peak_kwh']:,} kWh" if data['peak_kwh'] else "  Peak: Not found")
        print(f"  Off-Peak: {data['off_peak_kwh']:,} kWh" if data['off_peak_kwh'] else "  Off-Peak: Not found")
        
        if data['meter_readings']:
            print(f"\nMeter Readings:")
            for meter in data['meter_readings']:
                usage = meter['current_reading'] - meter['previous_reading']
                print(f"  {meter['meter_id']} ({meter['service_type']}): {usage:,} kWh")
        
        if data['charges']:
            print(f"\nCharges:")
            for charge in data['charges']:
                print(f"  {charge['description']}: ${charge['amount']:,.2f} (${charge['rate']}/kWh)")
        
        # Save as JSON
        with open("extracted_utility_data.json", "w") as f:
            json.dump(data, f, indent=2)
        print(f"\nFull data saved to: extracted_utility_data.json")
        
        # Calculate emissions (example factor: 0.4 kg CO2 per kWh)
        if data['total_kwh']:
            emissions = data['total_kwh'] * 0.4
            print(f"\nEstimated CO2 Emissions: {emissions:,.2f} kg")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()