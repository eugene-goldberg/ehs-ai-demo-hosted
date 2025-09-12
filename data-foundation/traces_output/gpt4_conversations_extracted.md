

**Human Input:**

You are an expert at extracting structured data from invoices,
especially those related to environmental services, waste management, and utilities.
Extract all relevant information and return it as JSON.
Pay special attention to:
- Invoice number and dates
- Vendor information
- Line items (especially environmental services)
- Waste disposal and recycling items
- Environmental fees
- Total amounts
        For dates, use YYYY-MM-DD format.
        For missing values, use null.
        
        The output should be formatted as a JSON instance that conforms to the JSON schema below.
As an example, for the schema {"properties": {"foo": {"title": "Foo", "description": "a list of strings", "type": "array", "items": {"type": "string"}}}, "required": ["foo"]}
the object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema. The object {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.
Here is the output schema:
{"description": "Structured data for supplier invoices.", "properties": {"invoice_number": {"anyOf": [{"type": "string"}, {"type": "null"}], "description": "Invoice number", "title": "Invoice Number"}, "invoice_date": {"anyOf": [{"type": "string"}, {"type": "null"}], "description": "Invoice date (YYYY-MM-DD)", "title": "Invoice Date"}, "due_date": {"anyOf": [{"type": "string"}, {"type": "null"}], "description": "Payment due date (YYYY-MM-DD)", "title": "Due Date"}, "vendor_name": {"anyOf": [{"type": "string"}, {"type": "null"}], "description": "Vendor/supplier name", "title": "Vendor Name"}, "vendor_address": {"anyOf": [{"type": "string"}, {"type": "null"}], "description": "Vendor address", "title": "Vendor Address"}, "line_items": {"anyOf": [{"items": {"additionalProperties": true, "type": "object"}, "type": "array"}, {"type": "null"}], "description": "List of line items with description, quantity, unit price, total", "title": "Line Items"}, "waste_disposal_items": {"anyOf": [{"items": {"additionalProperties": true, "type": "object"}, "type": "array"}, {"type": "null"}], "description": "Waste disposal related line items", "title": "Waste Disposal Items"}, "recycling_items": {"anyOf": [{"items": {"additionalProperties": true, "type": "object"}, "type": "array"}, {"type": "null"}], "description": "Recycling related line items", "title": "Recycling Items"}, "environmental_fees": {"anyOf": [{"type": "number"}, {"type": "null"}], "description": "Environmental or regulatory fees", "title": "Environmental Fees"}, "subtotal": {"anyOf": [{"type": "number"}, {"type": "null"}], "description": "Subtotal before tax", "title": "Subtotal"}, "tax_amount": {"anyOf": [{"type": "number"}, {"type": "null"}], "description": "Tax amount", "title": "Tax Amount"}, "total_amount": {"anyOf": [{"type": "number"}, {"type": "null"}], "description": "Total invoice amount", "title": "Total Amount"}}, "required": ["invoice_number", "invoice_date", "due_date", "vendor_name", "vendor_address", "line_items", "waste_disposal_items", "recycling_items", "environmental_fees", "subtotal", "tax_amount", "total_amount"]}

> Please classify this document:
> 
> CONTENT:
> Here is the structured data extracted from the provided text:
> 
> ### Tables with Numerical Data
> 
> **Electricity Usage Details**
> | Meter ID     | Service Type              | Previous Reading | Current Reading |
> |--------------|---------------------------|------------------|------------------|
> | MTR-7743-A   | Commercial - Peak         | 543210 kWh       | 613210 kWh       |
> | MTR-7743-B   | Commercial - Off-Peak     | 891234 kWh       | 951234 kWh       |
> | **Total Consumption** |                   |                  | **130,000 kWh**  |
> 
> **Current Charges Breakdown**
> | Description                                   | Rate              | Amount      |
> |-----------------------------------------------|-------------------|-------------|
> | Peak Demand Usage (70,000 kWh)               | $0.1255 / kWh     | $8,785.00   |
> | Off-Peak Demand Usage (60,000 kWh)           | $0.0890 / kWh     | $5,340.00   |
> | Base Service Charge                           |                   | $250.00     |
> | State Environmental Surcharge                 |                   | $875.50     |
> | Grid Infrastructure Fee                        |                   | $182.39     |
> | **Total Current Charges**                     |                   | **$15,432.89** |
> 
> ### Dates and Time Periods
> - Statement Date: July 5, 2025
> - Due Date: August 1, 2025
> - Billing Period: June 1, 2025 to June 30, 2025
> 
> ### Quantities and Measurements
> - Total Consumption: 130,000 kWh
> - Peak Demand Usage: 70,000 kWh
> - Off-Peak Demand Usage: 60,000 kWh
> 
> ### Costs and Financial Information
> - Amount Due: $15,432.89
> - Peak Demand Usage Charge: $8,785.00
> - Off-Peak Demand Usage Charge: $5,340.00
> - Base Service Charge: $250.00
> - State Environmental Surcharge: $875.50
> - Grid Infrastructure Fee: $182.39
> 
> ### Compliance-related Information
> - State Environmental Surcharge: $875.50 (implies compliance with environmental regulations)
> 
> ### Environmental Metrics
> - State Environmental Surcharge: $875.50 (indicates a charge related to environmental impact)
> 
> This structured data provides a clear overview of the utility statement for Apex Manufacturing Inc.
> 
> EXTRACTED FEATURES:
> - Tables found: True
> - Key terms: bill, electricity, statement, peak demand, consumption, usage, electric, utility, kwh, meter
> - Page count: 1

**AI Response:**

```json
{"document_type": "electricity_bill", "confidence": 1.0, "reasoning": "The document contains details about electricity usage, meter readings, and charges related to electricity consumption, which are typical features of an electricity bill."}
```

---

### Conversation 2

**Metadata:**
- Run ID: `65c6f113-29b2-4232-9ec3-7b4e7942ccce`
- Session ID: `133b8879-3c32-4d23-882d-b35bbab722eb`
- Status: `success`

**Human Input:**

> Please classify this document:
> 
> CONTENT:
> Here is the structured data extracted from the provided text:
> 
> ### Tables with Numerical Data
> 1. **Water Usage Details**
> | Meter ID      | Service Type | Previous Read   | Current Read    | Usage        |
> |---------------|--------------|------------------|------------------|--------------|
> | WTR-MAIN-01   | Commercial    | 8,765,400 gal    | 9,015,400 gal    | 250,000 gal  |
> 
> 2. **Current Charges Breakdown**
> | Description                                                       | Rate                | Amount      |
> |-------------------------------------------------------------------|---------------------|-------------|
> | Water Consumption (250,000 Gallons)                               | $0.0150 / Gallon    | $3,750.00   |
> | Sewer Service Charge (Based on water usage)                       |                     | $950.00     |
> | Stormwater Management Fee                                          |                     | $125.00     |
> | State Water Conservation Tax                                        |                     | $61.50      |
> | Infrastructure Improvement Surcharge                               |                     | $5.00       |
> | **Total Current Charges**                                          |                     | **$4,891.50** |
> 
> ### Dates and Time Periods
> - Statement Date: **July 6, 2025**
> - Due Date: **August 5, 2025**
> - Billing Period: **June 1, 2025 to June 30, 2025**
> 
> ### Quantities and Measurements
> - Water Usage: **250,000 gallons**
> - Previous Meter Read: **8,765,400 gallons**
> - Current Meter Read: **9,015,400 gallons**
> 
> ### Costs and Financial Information
> - Amount Due: **$4,891.50**
> - Water Consumption Charge: **$3,750.00**
> - Sewer Service Charge: **$950.00**
> - Stormwater Management Fee: **$125.00**
> - State Water Conservation Tax: **$61.50**
> - Infrastructure Improvement Surcharge: **$5.00**
> 
> ### Compliance-related Information
> - State Water Conservation Tax indicates compliance with state regulations regarding water usage.
> 
> ### Environmental Metrics
> - Water Consumption: **250,000 gallons** (indicates the amount of water used, relevant for environmental impact assessments).
> 
> This structured data provides a clear overview of the water utility statement for Apex Manufacturing Inc.
> 
> EXTRACTED FEATURES:
> - Tables found: True
> - Key terms: bill, water utility, stormwater, statement, gallons, consumption, sewer, usage, utility, water
> - Page count: 1

**AI Response:**

```json
{"document_type": "water_bill", "confidence": 1.0, "reasoning": "The document contains detailed information about water usage, sewer charges, and meter readings, which are typical contents of a water bill."}
```

---

### Conversation 3

**Metadata:**
- Run ID: `7a05f63d-fda3-43a6-85fb-829a5be1cfa9`
- Session ID: `133b8879-3c32-4d23-882d-b35bbab722eb`
- Status: `success`

**Human Input:**

> Please classify this document:
> 
> CONTENT:
> Here is the structured data extracted from the provided text:
> 
> ### Tables with Numerical Data
> | Line Item | Container Type          | Quantity | Description                                           |
> |-----------|-------------------------|----------|-------------------------------------------------------|
> | 1         | Open Top Roll-off       | 1        | Industrial Solid Waste - Mixed non-recyclable production scrap (plastic, wood, fabric) |
> 
> ### Dates and Time Periods
> - Generator Certification Date: **07/15/2025**
> - Transporter Acknowledgment Date: **07/1** (incomplete, likely intended to be **07/15/2025**)
> 
> ### Quantities and Measurements
> - Quantity of waste: **1** (Open Top Roll-off)
> 
> ### Costs and Financial Information
> - Purchase Order Number: **98765**
> 
> ### Compliance-related Information
> - Generator EPA ID: **CAL000123456**
> - Facility EPA ID: **CAL000654321**
> - Declaration of compliance with international and national governmental regulations.
> 
> ### Environmental Metrics
> - Waste Type: **Industrial Solid Waste - Mixed non-recyclable production scrap (plastic, wood, fabric)**
> - Waste Classification: **Non-hazardous solid waste**
> 
> ### Contact Information
> - **Generator/Shipper:**
> - Name: Apex Manufacturing Inc.
> - Address: 789 Production Way, Mechanicsburg, CA 93011
> - Contact: John Doe, EHS Manager
> - Phone: (555) 123-7890
> 
> - **Transporter:**
> - Carrier Name: Evergreen Environmental
> - Address: 333 Disposal Rd, Greenville, CA 91101
> - Phone: (555) 555-1234
> - Vehicle ID: Truck 72
> - Driver: M. Smith
> 
> - **Receiving Facility:**
> - Name: Green Valley Landfill
> - Address: 100 Landfill Lane, Greenville, CA 91102
> 
> This structured data provides a clear overview of the waste manifest details, including compliance, contact information, and waste characteristics.
> 
> EXTRACTED FEATURES:
> - Tables found: True
> - Key terms: manifest, disposal, generator, waste, facility, transporter, hazardous, epa id, certification
> - Page count: 1

**AI Response:**

```json
{"document_type": "waste_manifest", "confidence": 1.0, "reasoning": "The document contains information about waste type, generator/shipper, transporter, and receiving facility, which are typical features of a waste manifest document."}
```

---

### Conversation 4

**Metadata:**
- Run ID: `6a6c3e95-3feb-4a20-baa2-5a65521445be`
- Session ID: `133b8879-3c32-4d23-882d-b35bbab722eb`
- Status: `success`

**Human Input:**

> Please classify this document:
> 
> CONTENT:
> Here is the structured data extracted from the provided text:
> 
> ### Tables with Numerical Data
> 
> **Electricity Usage Details**
> | Meter ID     | Service Type              | Previous Reading | Current Reading |
> |--------------|---------------------------|------------------|------------------|
> | MTR-7743-A   | Commercial - Peak         | 543210 kWh       | 613210 kWh       |
> | MTR-7743-B   | Commercial - Off-Peak     | 891234 kWh       | 951234 kWh       |
> | **Total Consumption** |                   |                  | **130,000 kWh**  |
> 
> **Current Charges Breakdown**
> | Description                                   | Rate              | Amount      |
> |-----------------------------------------------|-------------------|-------------|
> | Peak Demand Usage (70,000 kWh)               | $0.1255 / kWh     | $8,785.00   |
> | Off-Peak Demand Usage (60,000 kWh)           | $0.0890 / kWh     | $5,340.00   |
> | Base Service Charge                           |                   | $250.00     |
> | State Environmental Surcharge                 |                   | $875.50     |
> | Grid Infrastructure Fee                        |                   | $182.39     |
> | **Total Current Charges**                     |                   | **$15,432.89** |
> 
> ### Dates and Time Periods
> - Statement Date: July 5, 2025
> - Due Date: August 1, 2025
> - Billing Period: June 1, 2025 to June 30, 2025
> 
> ### Quantities and Measurements
> - Total Consumption: 130,000 kWh
> - Peak Demand Usage: 70,000 kWh
> - Off-Peak Demand Usage: 60,000 kWh
> 
> ### Costs and Financial Information
> - Amount Due: $15,432.89
> - Peak Demand Usage Charge: $8,785.00
> - Off-Peak Demand Usage Charge: $5,340.00
> - Base Service Charge: $250.00
> - State Environmental Surcharge: $875.50
> - Grid Infrastructure Fee: $182.39
> 
> ### Compliance-related Information
> - State Environmental Surcharge: $875.50 (indicates compliance with environmental regulations)
> 
> ### Environmental Metrics
> - State Environmental Surcharge: $875.50 (could be related to environmental compliance costs)
> 
> This structured data provides a clear overview of the utility statement for Apex Manufacturing Inc.
> 
> EXTRACTED FEATURES:
> - Tables found: True
> - Key terms: electric, usage, statement, consumption, utility, peak demand, bill, kwh, electricity, meter
> - Page count: 1

**AI Response:**

```json
{"document_type": "electricity_bill", "confidence": 1.0, "reasoning": "The document contains details about electricity usage, meter readings, and charges related to electricity consumption, which are typical features of an electricity bill."}
```
