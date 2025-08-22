# Waste Manifest RAG Implementation Plan

Based on the waste manifest PDF analysis, here's a comprehensive plan to extend the ingestion workflow for waste manifest documents:

## Waste Manifest Structure Analysis

### Key Data Elements Found:

1. **Manifest Tracking Information**
   - Manifest Tracking Number: "EES-2025-0715-A45"
   - Manifest Type: "Hazardous Waste Manifest" 
   - Issue Date: "July 15, 2025"
   - Document Status: "Original"

2. **Generator Information**
   - Company Name: "Apex Manufacturing Inc."
   - EPA ID Number: "CAL000123456"
   - Contact Person: "John Doe, Environmental Coordinator"
   - Phone: "(800) 555-0147"
   - Address: "789 Production Way, Mechanicsburg, CA 93011"

3. **Transporter Information**
   - Company Name: "Evergreen Environmental Services"
   - EPA ID Number: "CAL000789012"
   - Vehicle ID: "Truck 72"
   - Driver Name: "M. Smith"
   - Driver License: "CA-CDL-45789"

4. **Receiving Facility Information**
   - Company Name: "Green Valley Landfill"
   - EPA ID Number: "CAL000654321"
   - Contact Person: "Sarah Johnson"
   - Phone: "(800) 555-0203"
   - Address: "123 Disposal Drive, Riverside, CA 92503"

5. **Waste Description**
   - Waste Description: "Industrial Solid Waste - Mixed non-recyclable production scrap"
   - Container Type: "Open Top Roll-off"
   - Container Quantity: 1
   - Total Quantity: "15 tons"
   - Unit: "tons"
   - Waste Classification: "Non-hazardous industrial solid waste"

6. **Certification and Signatures**
   - Generator Certification: Signed by John Doe on July 15, 2025
   - Transporter Acknowledgment: Signed by M. Smith on July 15, 2025
   - Facility Operator Certification: Signed by Sarah Johnson on July 16, 2025

7. **Special Handling Instructions**
   - Instructions: "No special handling required. Standard landfill disposal procedures apply."

## Implementation Plan

### 1. Create WasteManifestData Model (ehs_extractors.py)

```python
class WasteManifestData(BaseModel):
    """Structured data for waste manifests."""
    # Manifest Information
    manifest_tracking_number: Optional[str] = Field(description="Unique manifest tracking number")
    manifest_type: Optional[str] = Field(description="Type of waste manifest (hazardous, non-hazardous)")
    issue_date: Optional[str] = Field(description="Date manifest was issued (YYYY-MM-DD)")
    document_status: Optional[str] = Field(description="Document status (original, copy, etc.)")
    
    # Generator Information
    generator_name: Optional[str] = Field(description="Waste generator company name")
    generator_epa_id: Optional[str] = Field(description="Generator EPA ID number")
    generator_contact_person: Optional[str] = Field(description="Generator contact person name and title")
    generator_phone: Optional[str] = Field(description="Generator phone number")
    generator_address: Optional[str] = Field(description="Generator company address")
    
    # Transporter Information
    transporter_name: Optional[str] = Field(description="Waste transporter company name")
    transporter_epa_id: Optional[str] = Field(description="Transporter EPA ID number")
    vehicle_id: Optional[str] = Field(description="Transport vehicle identification")
    driver_name: Optional[str] = Field(description="Driver name")
    driver_license: Optional[str] = Field(description="Driver license number")
    
    # Receiving Facility Information
    facility_name: Optional[str] = Field(description="Receiving facility name")
    facility_epa_id: Optional[str] = Field(description="Facility EPA ID number")
    facility_contact_person: Optional[str] = Field(description="Facility contact person")
    facility_phone: Optional[str] = Field(description="Facility phone number")
    facility_address: Optional[str] = Field(description="Facility address")
    
    # Waste Line Items (supporting multiple waste types)
    waste_items: Optional[List[Dict[str, Any]]] = Field(
        description="List of waste items with description, container_type, quantity, unit, classification"
    )
    
    # Certifications
    generator_certification_date: Optional[str] = Field(description="Date generator certified (YYYY-MM-DD)")
    generator_signature: Optional[str] = Field(description="Generator signature/name")
    transporter_acknowledgment_date: Optional[str] = Field(description="Date transporter acknowledged (YYYY-MM-DD)")
    transporter_signature: Optional[str] = Field(description="Transporter signature/name")
    facility_certification_date: Optional[str] = Field(description="Date facility certified (YYYY-MM-DD)")
    facility_signature: Optional[str] = Field(description="Facility signature/name")
    
    # Special Handling
    special_handling_instructions: Optional[str] = Field(description="Special handling or disposal instructions")
    
    # Calculated Fields
    total_waste_quantity: Optional[float] = Field(description="Total quantity of all waste items")
    total_waste_unit: Optional[str] = Field(description="Unit for total waste quantity")
```

### 2. Create WasteManifestExtractor Class

```python
class WasteManifestExtractor(BaseExtractor):
    """Extract data from waste manifests."""
    
    def __init__(self, llm_model: str = "gpt-4"):
        super().__init__(llm_model)
        
        # Create output parser
        self.parser = JsonOutputParser(pydantic_object=WasteManifestData)
        
        # Create prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at extracting structured data from waste manifests.
            Extract all relevant information and return it as JSON.
            Pay special attention to:
            - Manifest tracking number and document details
            - All parties involved (generator, transporter, facility) with EPA IDs
            - Complete waste descriptions including quantities and classifications
            - All certification dates and signatures
            - Special handling instructions
            
            For waste_items, extract as a list with objects containing:
            - description: Full waste description
            - container_type: Type of container used
            - quantity: Numeric quantity
            - unit: Unit of measurement (tons, cubic yards, gallons, etc.)
            - classification: Waste classification (hazardous, non-hazardous, etc.)
            
            For dates, convert to YYYY-MM-DD format.
            For missing values, use null.
            
            {format_instructions}"""),
            ("human", "Extract data from this waste manifest:\n\n{content}")
        ])
    
    def extract(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Extract structured data from waste manifest content.
        
        Args:
            content: Text content of the waste manifest
            metadata: Optional metadata about the document
            
        Returns:
            Dictionary of extracted data
        """
        try:
            # Format prompt
            formatted_prompt = self.prompt.format_messages(
                format_instructions=self.parser.get_format_instructions(),
                content=content
            )
            
            # Get LLM response
            response = self.llm.invoke(formatted_prompt)
            
            # Parse response
            extracted_data = self.parser.parse(response.content)
            
            # Add metadata
            if metadata:
                extracted_data["metadata"] = metadata
            
            logger.info(f"Successfully extracted waste manifest data")
            return extracted_data
            
        except Exception as e:
            logger.error(f"Error extracting waste manifest data: {str(e)}")
            # Return empty structure on error
            return WasteManifestData().dict()
```

### 3. Neo4j Schema Design

#### Node Types:
```cypher
// Core document node
CREATE CONSTRAINT waste_manifest_id IF NOT EXISTS FOR (d:Document) REQUIRE d.document_id IS UNIQUE;

// Waste shipment tracking
CREATE CONSTRAINT manifest_tracking_number IF NOT EXISTS FOR (w:WasteShipment) REQUIRE w.tracking_number IS UNIQUE;

// EPA entities
CREATE CONSTRAINT epa_id IF NOT EXISTS FOR (g:WasteGenerator) REQUIRE g.epa_id IS UNIQUE;
CREATE CONSTRAINT transporter_epa_id IF NOT EXISTS FOR (t:Transporter) REQUIRE t.epa_id IS UNIQUE;
CREATE CONSTRAINT facility_epa_id IF NOT EXISTS FOR (f:DisposalFacility) REQUIRE f.epa_id IS UNIQUE;
```

#### Node Labels and Properties:

1. **Document:WasteManifest**
   ```cypher
   Properties: document_id, file_path, upload_date, manifest_type, issue_date, document_status
   ```

2. **WasteShipment**
   ```cypher
   Properties: tracking_number, issue_date, total_quantity, total_unit, status, special_instructions
   ```

3. **WasteGenerator**
   ```cypher
   Properties: epa_id, name, contact_person, phone, address, certification_date, signature
   ```

4. **Transporter**
   ```cypher
   Properties: epa_id, name, vehicle_id, driver_name, driver_license, acknowledgment_date, signature
   ```

5. **DisposalFacility**
   ```cypher
   Properties: epa_id, name, contact_person, phone, address, certification_date, signature
   ```

6. **WasteItem**
   ```cypher
   Properties: description, container_type, quantity, unit, classification
   ```

#### Relationships:

```cypher
// Document relationships
(:Document:WasteManifest)-[:TRACKS]->(:WasteShipment)

// Shipment chain
(:WasteShipment)-[:GENERATED_BY]->(:WasteGenerator)
(:WasteShipment)-[:TRANSPORTED_BY]->(:Transporter)
(:WasteShipment)-[:DISPOSED_AT]->(:DisposalFacility)
(:WasteShipment)-[:CONTAINS_WASTE]->(:WasteItem)

// Temporal relationships
(:WasteGenerator)-[:CERTIFIED_ON {date: "2025-07-15"}]->(:WasteShipment)
(:Transporter)-[:ACKNOWLEDGED_ON {date: "2025-07-15"}]->(:WasteShipment)
(:DisposalFacility)-[:RECEIVED_ON {date: "2025-07-16"}]->(:WasteShipment)
```

### 4. Document Type Detection Update

Update the document classification logic to detect waste manifests:

```python
# In ingestion_workflow.py or document classifier
WASTE_MANIFEST_INDICATORS = [
    "manifest tracking number",
    "epa id number",
    "waste generator",
    "transporter", 
    "disposal facility",
    "hazardous waste manifest",
    "waste description",
    "container type"
]

def classify_document_type(content: str) -> str:
    """Classify document type based on content."""
    content_lower = content.lower()
    
    # Existing classifications...
    
    # Check for waste manifest
    manifest_score = sum(1 for indicator in WASTE_MANIFEST_INDICATORS 
                        if indicator in content_lower)
    if manifest_score >= 4:
        return "waste_manifest"
    
    return "unknown"
```

### 5. Integration into Existing Workflow

Update the ingestion workflow to handle waste manifests:

```python
# In ingestion_workflow.py
from ..extractors.ehs_extractors import WasteManifestExtractor

class IngestionWorkflow:
    def __init__(self, ...):
        # Existing initialization...
        self.extractors = {
            "utility_bill": UtilityBillExtractor(llm_model),
            "water_bill": WaterBillExtractor(llm_model),
            "permit": PermitExtractor(llm_model),
            "invoice": InvoiceExtractor(llm_model),
            "waste_manifest": WasteManifestExtractor(llm_model)  # New
        }
```

### 6. Neo4j Graph Creation Logic

```python
def create_waste_manifest_graph(self, extracted_data: Dict[str, Any], document_id: str) -> Tuple[List[Dict], List[Dict]]:
    """Create Neo4j nodes and relationships for waste manifest data."""
    
    nodes = []
    relationships = []
    
    # Create WasteShipment node
    shipment_node = {
        "labels": ["WasteShipment"],
        "properties": {
            "tracking_number": extracted_data.get("manifest_tracking_number"),
            "issue_date": extracted_data.get("issue_date"),
            "total_quantity": extracted_data.get("total_waste_quantity"),
            "total_unit": extracted_data.get("total_waste_unit"),
            "special_instructions": extracted_data.get("special_handling_instructions")
        }
    }
    nodes.append(shipment_node)
    
    # Create WasteGenerator node
    if extracted_data.get("generator_epa_id"):
        generator_node = {
            "labels": ["WasteGenerator"],
            "properties": {
                "epa_id": extracted_data.get("generator_epa_id"),
                "name": extracted_data.get("generator_name"),
                "contact_person": extracted_data.get("generator_contact_person"),
                "phone": extracted_data.get("generator_phone"),
                "address": extracted_data.get("generator_address"),
                "certification_date": extracted_data.get("generator_certification_date"),
                "signature": extracted_data.get("generator_signature")
            }
        }
        nodes.append(generator_node)
        
        # Create relationship
        relationships.append({
            "type": "GENERATED_BY",
            "source_labels": ["WasteShipment"],
            "source_properties": {"tracking_number": extracted_data.get("manifest_tracking_number")},
            "target_labels": ["WasteGenerator"],
            "target_properties": {"epa_id": extracted_data.get("generator_epa_id")},
            "properties": {}
        })
    
    # Create Transporter node
    if extracted_data.get("transporter_epa_id"):
        transporter_node = {
            "labels": ["Transporter"],
            "properties": {
                "epa_id": extracted_data.get("transporter_epa_id"),
                "name": extracted_data.get("transporter_name"),
                "vehicle_id": extracted_data.get("vehicle_id"),
                "driver_name": extracted_data.get("driver_name"),
                "driver_license": extracted_data.get("driver_license"),
                "acknowledgment_date": extracted_data.get("transporter_acknowledgment_date"),
                "signature": extracted_data.get("transporter_signature")
            }
        }
        nodes.append(transporter_node)
        
        # Create relationship
        relationships.append({
            "type": "TRANSPORTED_BY",
            "source_labels": ["WasteShipment"],
            "source_properties": {"tracking_number": extracted_data.get("manifest_tracking_number")},
            "target_labels": ["Transporter"],
            "target_properties": {"epa_id": extracted_data.get("transporter_epa_id")},
            "properties": {}
        })
    
    # Create DisposalFacility node
    if extracted_data.get("facility_epa_id"):
        facility_node = {
            "labels": ["DisposalFacility"],
            "properties": {
                "epa_id": extracted_data.get("facility_epa_id"),
                "name": extracted_data.get("facility_name"),
                "contact_person": extracted_data.get("facility_contact_person"),
                "phone": extracted_data.get("facility_phone"),
                "address": extracted_data.get("facility_address"),
                "certification_date": extracted_data.get("facility_certification_date"),
                "signature": extracted_data.get("facility_signature")
            }
        }
        nodes.append(facility_node)
        
        # Create relationship
        relationships.append({
            "type": "DISPOSED_AT",
            "source_labels": ["WasteShipment"],
            "source_properties": {"tracking_number": extracted_data.get("manifest_tracking_number")},
            "target_labels": ["DisposalFacility"],
            "target_properties": {"epa_id": extracted_data.get("facility_epa_id")},
            "properties": {}
        })
    
    # Create WasteItem nodes
    waste_items = extracted_data.get("waste_items", [])
    for i, item in enumerate(waste_items):
        waste_item_node = {
            "labels": ["WasteItem"],
            "properties": {
                "item_id": f"{extracted_data.get('manifest_tracking_number')}_item_{i}",
                "description": item.get("description"),
                "container_type": item.get("container_type"),
                "quantity": item.get("quantity"),
                "unit": item.get("unit"),
                "classification": item.get("classification")
            }
        }
        nodes.append(waste_item_node)
        
        # Create relationship
        relationships.append({
            "type": "CONTAINS_WASTE",
            "source_labels": ["WasteShipment"],
            "source_properties": {"tracking_number": extracted_data.get("manifest_tracking_number")},
            "target_labels": ["WasteItem"],
            "target_properties": {"item_id": f"{extracted_data.get('manifest_tracking_number')}_item_{i}"},
            "properties": {}
        })
    
    return nodes, relationships
```

### 7. Emission Calculation Approach

Extend the emission calculation system to handle waste-related emissions:

```python
class WasteEmissionCalculator:
    """Calculate emissions from waste disposal activities."""
    
    # EPA emission factors (example values)
    LANDFILL_METHANE_FACTOR = 0.5  # metric tons CO2e per ton of waste
    TRANSPORTATION_FACTOR = 0.089  # kg CO2e per ton-mile
    
    def calculate_disposal_emissions(self, waste_quantity: float, waste_unit: str, disposal_method: str = "landfill") -> float:
        """Calculate emissions from waste disposal."""
        
        # Convert to standard unit (metric tons)
        tons = self.convert_to_metric_tons(waste_quantity, waste_unit)
        
        if disposal_method.lower() == "landfill":
            return tons * self.LANDFILL_METHANE_FACTOR
        elif disposal_method.lower() == "recycling":
            return tons * -0.2  # Negative emissions for recycling benefit
        else:
            return tons * 0.3  # Generic disposal factor
    
    def calculate_transportation_emissions(self, waste_quantity: float, waste_unit: str, distance_miles: float) -> float:
        """Calculate emissions from waste transportation."""
        
        tons = self.convert_to_metric_tons(waste_quantity, waste_unit)
        ton_miles = tons * distance_miles
        
        return ton_miles * self.TRANSPORTATION_FACTOR / 1000  # Convert kg to metric tons
    
    def convert_to_metric_tons(self, quantity: float, unit: str) -> float:
        """Convert various units to metric tons."""
        
        conversion_factors = {
            "tons": 0.907185,  # US tons to metric tons
            "pounds": 0.000453592,
            "kilograms": 0.001,
            "cubic_yards": 0.3,  # Estimated based on typical waste density
            "gallons": 0.00378541 * 0.8  # Gallons to cubic meters * typical liquid waste density
        }
        
        return quantity * conversion_factors.get(unit.lower(), 1.0)
```

### 8. Validation Rules

Implement validation for waste manifest data:

```python
def validate_waste_manifest_data(extracted_data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate extracted waste manifest data."""
    
    validation_results = {
        "is_valid": True,
        "errors": [],
        "warnings": []
    }
    
    # Required fields validation
    required_fields = [
        "manifest_tracking_number",
        "generator_epa_id", 
        "transporter_epa_id",
        "facility_epa_id",
        "waste_items"
    ]
    
    for field in required_fields:
        if not extracted_data.get(field):
            validation_results["errors"].append(f"Missing required field: {field}")
            validation_results["is_valid"] = False
    
    # EPA ID format validation
    epa_id_pattern = r"^[A-Z]{2,3}\d{9,12}$"
    epa_fields = ["generator_epa_id", "transporter_epa_id", "facility_epa_id"]
    
    for field in epa_fields:
        epa_id = extracted_data.get(field)
        if epa_id and not re.match(epa_id_pattern, epa_id):
            validation_results["warnings"].append(f"Invalid EPA ID format: {field}")
    
    # Date validation
    date_fields = ["issue_date", "generator_certification_date", "transporter_acknowledgment_date", "facility_certification_date"]
    
    for field in date_fields:
        date_value = extracted_data.get(field)
        if date_value:
            try:
                datetime.strptime(date_value, "%Y-%m-%d")
            except ValueError:
                validation_results["errors"].append(f"Invalid date format for {field}: {date_value}")
                validation_results["is_valid"] = False
    
    # Waste items validation
    waste_items = extracted_data.get("waste_items", [])
    if len(waste_items) == 0:
        validation_results["errors"].append("No waste items found")
        validation_results["is_valid"] = False
    
    for i, item in enumerate(waste_items):
        if not item.get("description"):
            validation_results["errors"].append(f"Waste item {i+1} missing description")
            validation_results["is_valid"] = False
        if not item.get("quantity"):
            validation_results["warnings"].append(f"Waste item {i+1} missing quantity")
    
    return validation_results
```

### 9. Testing Approach

#### Sample Test Data (from PDF):
```python
SAMPLE_WASTE_MANIFEST_DATA = {
    "manifest_tracking_number": "EES-2025-0715-A45",
    "manifest_type": "Hazardous Waste Manifest",
    "issue_date": "2025-07-15",
    "document_status": "Original",
    
    "generator_name": "Apex Manufacturing Inc.",
    "generator_epa_id": "CAL000123456",
    "generator_contact_person": "John Doe, Environmental Coordinator",
    "generator_phone": "(800) 555-0147",
    "generator_address": "789 Production Way, Mechanicsburg, CA 93011",
    "generator_certification_date": "2025-07-15",
    "generator_signature": "John Doe",
    
    "transporter_name": "Evergreen Environmental Services",
    "transporter_epa_id": "CAL000789012",
    "vehicle_id": "Truck 72",
    "driver_name": "M. Smith",
    "driver_license": "CA-CDL-45789",
    "transporter_acknowledgment_date": "2025-07-15",
    "transporter_signature": "M. Smith",
    
    "facility_name": "Green Valley Landfill",
    "facility_epa_id": "CAL000654321",
    "facility_contact_person": "Sarah Johnson",
    "facility_phone": "(800) 555-0203",
    "facility_address": "123 Disposal Drive, Riverside, CA 92503",
    "facility_certification_date": "2025-07-16",
    "facility_signature": "Sarah Johnson",
    
    "waste_items": [
        {
            "description": "Industrial Solid Waste - Mixed non-recyclable production scrap",
            "container_type": "Open Top Roll-off",
            "quantity": 15.0,
            "unit": "tons",
            "classification": "Non-hazardous industrial solid waste"
        }
    ],
    
    "special_handling_instructions": "No special handling required. Standard landfill disposal procedures apply.",
    "total_waste_quantity": 15.0,
    "total_waste_unit": "tons"
}
```

#### Expected Node/Relationship Counts:
- **Nodes**: 5 (Document, WasteShipment, WasteGenerator, Transporter, DisposalFacility, WasteItem)
- **Relationships**: 4 (TRACKS, GENERATED_BY, TRANSPORTED_BY, DISPOSED_AT, CONTAINS_WASTE)

#### Test Queries:
```cypher
// Verify waste shipment chain
MATCH (d:Document:WasteManifest)-[:TRACKS]->(ws:WasteShipment)
-[:GENERATED_BY]->(g:WasteGenerator)
MATCH (ws)-[:TRANSPORTED_BY]->(t:Transporter)
MATCH (ws)-[:DISPOSED_AT]->(f:DisposalFacility)
MATCH (ws)-[:CONTAINS_WASTE]->(wi:WasteItem)
RETURN d.document_id, ws.tracking_number, g.name, t.name, f.name, wi.description

// Find all waste manifests for a generator
MATCH (g:WasteGenerator {epa_id: "CAL000123456"})<-[:GENERATED_BY]-(ws:WasteShipment)
RETURN g.name, count(ws) as total_shipments, sum(ws.total_quantity) as total_waste

// Calculate emissions for waste disposal
MATCH (ws:WasteShipment)-[:DISPOSED_AT]->(f:DisposalFacility)
MATCH (ws)-[:CONTAINS_WASTE]->(wi:WasteItem)
RETURN ws.tracking_number, f.name, wi.quantity * 0.5 as estimated_co2_emissions
```

### 10. File Updates Required

1. **ehs_extractors.py**: Add `WasteManifestData` model and `WasteManifestExtractor` class
2. **ingestion_workflow.py**: Update document type detection and add waste manifest handling
3. **document_indexer.py**: Add graph creation logic for waste manifests
4. **Update extract_ehs_data function**: Add "waste_manifest" to document type mapping

### 11. Integration Testing

Create integration tests to verify:

1. **PDF Processing**: Waste manifest PDF correctly parsed and content extracted
2. **Data Extraction**: All key fields properly extracted by LLM
3. **Validation**: Validation rules properly applied
4. **Graph Creation**: Correct nodes and relationships created in Neo4j
5. **Query Testing**: Sample queries return expected results
6. **Emission Calculations**: Proper CO2 equivalent calculations for waste disposal

This implementation plan provides a comprehensive framework for integrating waste manifest documents into the EHS AI RAG system, following the established patterns for utility bills and other document types while addressing the unique requirements of waste tracking and environmental compliance.