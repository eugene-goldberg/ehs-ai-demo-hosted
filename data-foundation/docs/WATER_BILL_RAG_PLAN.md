Water Bill Analysis and Implementation Plan

  Based on my careful examination of the water bill PDF, here's my comprehensive plan to extend the ingestion workflow:

  Water Bill Structure Analysis

  Key Data Elements Found:
  1. Provider Information
    - Name: "Aquaflow Municipal Water"
    - Type: Government utility (note the .gov website)
    - Address: "789 Reservoir Road, Clearwater, CA 90330"
    - Phone: "(800) 555-0211"
    - Website: "http://www.aquaflow.gov"
  2. Customer Information
    - Same as electric bill: "Apex Manufacturing Inc."
    - Billing address: "456 Industrial Ave. Mechanicsburg, CA 93011"
  3. Service Location
    - Same facility: "Apex Manufacturing - Plant A"
    - Address: "789 Production Way, Mechanicsburg, CA 93011"
  4. Account Details
    - Account Number: "44-19872-11-9"
    - Statement Date: "July 6, 2025"
    - Due Date: "August 5, 2025"
    - Billing Period: "June 1, 2025 to June 30, 2025"
  5. Water Usage
    - Total Usage: 250,000 gallons
    - Meter ID: "WTR-MAIN-01"
    - Service Type: "Commercial Water"
    - Previous Reading: 8,765,400 gal
    - Current Reading: 9,015,400 gal
  6. Charges Breakdown
    - Water Consumption: $3,750.00 ($0.0150/gallon)
    - Sewer Service Charge: $950.00
    - Stormwater Management Fee: $125.00
    - State Water Conservation Tax: $61.50
    - Infrastructure Improvement Surcharge: $5.00
    - Total: $4,891.50

  Implementation Plan

  1. Create WaterBillData Model (ehs_extractors.py)

  class WaterBillData(BaseModel):
      """Structured data for water bills."""
      # Dates
      billing_period_start: Optional[str] = Field(description="Start date of billing period (YYYY-MM-DD)")
      billing_period_end: Optional[str] = Field(description="End date of billing period (YYYY-MM-DD)")
      statement_date: Optional[str] = Field(description="Statement date (YYYY-MM-DD)")
      due_date: Optional[str] = Field(description="Payment due date (YYYY-MM-DD)")

      # Account and facility info
      account_number: Optional[str] = Field(description="Water utility account number")
      service_address: Optional[str] = Field(description="Service location address")
      facility_name: Optional[str] = Field(description="Name of facility receiving water service")
      facility_address: Optional[str] = Field(description="Full address of the facility")

      # Customer info
      customer_name: Optional[str] = Field(description="Customer name (company being billed)")
      customer_address: Optional[str] = Field(description="Customer billing address")

      # Utility provider info
      provider_name: Optional[str] = Field(description="Water utility provider name")
      provider_address: Optional[str] = Field(description="Water utility provider address")
      provider_phone: Optional[str] = Field(description="Provider phone number")
      provider_website: Optional[str] = Field(description="Provider website")
      provider_payment_address: Optional[str] = Field(description="Payment mailing address")

      # Water usage metrics
      total_gallons: Optional[float] = Field(description="Total water usage in gallons")
      total_cubic_meters: Optional[float] = Field(description="Total water usage in cubic meters (if provided)")

      # Costs
      total_cost: Optional[float] = Field(description="Total bill amount")
      water_consumption_cost: Optional[float] = Field(description="Cost for water consumption")
      sewer_service_charge: Optional[float] = Field(description="Sewer service charge")
      stormwater_fee: Optional[float] = Field(description="Stormwater management fee")
      conservation_tax: Optional[float] = Field(description="Water conservation tax")
      infrastructure_surcharge: Optional[float] = Field(description="Infrastructure improvement surcharge")

      # Meter readings
      meter_readings: Optional[List[Dict[str, Any]]] = Field(
          description="List of meter readings with meter_id, type, service_type, previous_reading, current_reading, usage, unit"
      )

      # Rate information
      water_rate: Optional[float] = Field(description="Rate per gallon or cubic meter")
      rate_unit: Optional[str] = Field(description="Unit for water rate (per gallon, per CCF, etc.)")

  2. Create WaterBillExtractor Class

  class WaterBillExtractor(BaseExtractor):
      """Extract data from water bills."""

      def __init__(self, llm_model: str = "gpt-4"):
          super().__init__(llm_model)
          self.parser = JsonOutputParser(pydantic_object=WaterBillData)

          self.prompt = ChatPromptTemplate.from_messages([
              ("system", """You are an expert at extracting structured data from water utility bills.
              Extract all relevant information and return it as JSON.
              Pay special attention to:
              - All dates (convert to YYYY-MM-DD format)
              - Distinguish between CUSTOMER and SERVICE LOCATION
              - Water usage in gallons (and cubic meters if provided)
              - All charges including sewer, stormwater, and taxes
              - Meter readings with units (gallons, CCF, cubic meters)
              - Provider information (may be municipal/government)
              
              For meter_readings, extract as a list with:
              - meter_id: The meter identifier
              - type: "water"
              - service_type: Type of water service
              - previous_reading: Previous meter reading
              - current_reading: Current meter reading
              - usage: The difference
              - unit: "gallons", "cubic_meters", or "CCF"
              
              {format_instructions}"""),
              ("human", "Extract data from this water bill:\n\n{content}")
          ])

  3. Update ingestion_workflow.py

  Add water bill transformation logic in transform_data method:

  elif doc_type == "water_bill":
      # Create water bill node
      bill_node = {
          "labels": ["WaterBill"],
          "properties": {
              "id": f"waterbill_{state['document_id']}",
              "billing_period_start": extracted.get("billing_period_start"),
              "billing_period_end": extracted.get("billing_period_end"),
              "total_gallons": extracted.get("total_gallons"),
              "total_cost": extracted.get("total_cost"),
              "water_consumption_cost": extracted.get("water_consumption_cost"),
              "sewer_service_charge": extracted.get("sewer_service_charge"),
              "due_date": extracted.get("due_date")
          }
      }
      nodes.append(bill_node)

      # Create relationships similar to utility bill
      # Document -> WaterBill
      # WaterBill -> Facility (reuse existing facility)
      # WaterBill -> Customer (reuse existing customer)
      # WaterBill -> WaterProvider (new provider node)
      # Meter -> Facility (water meter)
      # Meter -> WaterBill (RECORDED_IN)

      # Water emission calculation (if applicable)
      # Water treatment and distribution emissions
      # Typical factor: 0.0002 kg CO2/gallon

  4. Water-Specific Considerations

  1. Provider Type: Water utilities are often municipal/government entities
  2. Emission Calculations:
    - Water treatment/distribution emissions (~0.0002 kg CO2/gallon)
    - Wastewater treatment emissions (if sewer charges included)
  3. Units: Handle gallons, cubic meters, CCF (hundred cubic feet)
  4. Additional Charges: Sewer, stormwater, conservation taxes

  5. Graph Schema Additions

  // New node types
  (:WaterBill {id, billing_period_start/end, total_gallons, costs...})
  (:WaterProvider {id, name, address, phone, website, type: "municipal"})

  // New relationships
  (Document)-[:EXTRACTED_TO]->(WaterBill)
  (WaterBill)-[:BILLED_TO]->(Facility)
  (WaterBill)-[:BILLED_FOR]->(Customer)
  (WaterBill)-[:PROVIDED_BY]->(WaterProvider)
  (WaterMeter)-[:MONITORS]->(Facility)
  (WaterMeter)-[:RECORDED_IN]->(WaterBill)
  (WaterBill)-[:RESULTED_IN]->(Emission)  // If water emissions tracked

  6. Document Type Detection Update

  Update llama_parser.py to detect water bills:
  - Look for keywords: "water", "gallons", "sewer", "aqua"
  - Check for water-specific charges
  - Identify municipal provider patterns

  7. Testing Strategy

  1. Unit test for WaterBillExtractor
  2. Integration test with water_bill.pdf
  3. Verify all nodes and relationships created
  4. Test emission calculations
  5. Verify cross-referencing with existing facility

  Key Implementation Notes

  1. Reuse Existing Entities: Since it's the same facility and customer, link to existing nodes
  2. Municipal Provider: Handle government utilities differently (type field)
  3. Water Emissions: Optional but valuable for comprehensive carbon tracking
  4. Unit Conversion: Support multiple water measurement units
  5. Sewer Charges: Often based on water usage, important for total environmental impact

  This plan ensures comprehensive water bill data extraction while maintaining consistency with the existing electric bill implementation and
  reusing common entities where appropriate.