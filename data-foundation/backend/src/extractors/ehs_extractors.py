"""
EHS-specific data extractors using LLMs.
Extract structured data from utility bills, permits, invoices, and other EHS documents.
"""

import os
import logging
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
import re

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

try:
    from utils.llm_transcript_logger import log_llm_interaction
except ImportError:
    # If import fails, create a dummy function that does nothing
    def log_llm_interaction(role, content, context=None):
        pass

try:
    from utils.transcript_forwarder import forward_transcript_entry
except ImportError:
    # If import fails, create a no-op function
    def forward_transcript_entry(role, content, context=None):
        pass

logger = logging.getLogger(__name__)


# Pydantic models for structured output
class UtilityBillData(BaseModel):
    """Structured data for utility bills."""
    # Dates
    billing_period_start: Optional[str] = Field(description="Start date of billing period (YYYY-MM-DD)")
    billing_period_end: Optional[str] = Field(description="End date of billing period (YYYY-MM-DD)")
    statement_date: Optional[str] = Field(description="Statement/bill date (YYYY-MM-DD)")
    due_date: Optional[str] = Field(description="Payment due date (YYYY-MM-DD)")
    
    # Account and facility info
    account_number: Optional[str] = Field(description="Utility account number")
    service_address: Optional[str] = Field(description="Service address for the utility")
    facility_name: Optional[str] = Field(description="Name of the facility being billed")
    facility_address: Optional[str] = Field(description="Full address of the facility")
    
    # Customer info (billed to)
    customer_name: Optional[str] = Field(description="Customer name (company being billed)")
    customer_address: Optional[str] = Field(description="Customer billing address")
    customer_attention: Optional[str] = Field(description="Attention line (e.g., Accounts Payable)")
    
    # Utility provider info
    provider_name: Optional[str] = Field(description="Utility provider company name")
    provider_address: Optional[str] = Field(description="Utility provider address")
    provider_phone: Optional[str] = Field(description="Utility provider phone number")
    provider_website: Optional[str] = Field(description="Utility provider website")
    provider_payment_address: Optional[str] = Field(description="Payment mailing address")
    
    # Energy metrics
    total_kwh: Optional[float] = Field(description="Total kilowatt hours consumed")
    peak_kwh: Optional[float] = Field(description="Peak period kilowatt hours")
    off_peak_kwh: Optional[float] = Field(description="Off-peak period kilowatt hours")
    peak_demand_kw: Optional[float] = Field(description="Peak demand in kilowatts")
    
    # Other utilities
    gas_usage_therms: Optional[float] = Field(description="Natural gas usage in therms")
    gas_usage_ccf: Optional[float] = Field(description="Natural gas usage in CCF")
    water_usage_gallons: Optional[float] = Field(description="Water usage in gallons")
    water_usage_cubic_meters: Optional[float] = Field(description="Water usage in cubic meters")
    
    # Costs
    total_cost: Optional[float] = Field(description="Total bill amount")
    energy_cost: Optional[float] = Field(description="Cost for energy/electricity")
    demand_charges: Optional[float] = Field(description="Demand charges")
    
    # Charge breakdown
    base_service_charge: Optional[float] = Field(description="Base service charge")
    state_environmental_surcharge: Optional[float] = Field(description="State environmental surcharge")
    grid_infrastructure_fee: Optional[float] = Field(description="Grid infrastructure fee")
    charge_breakdown: Optional[List[Dict[str, Any]]] = Field(description="List of charge line items with description, rate, amount")
    
    # Additional info
    rate_schedule: Optional[str] = Field(description="Rate schedule or tariff")
    meter_readings: Optional[List[Dict[str, Any]]] = Field(description="List of meter readings with meter_id, type, service_type, previous_reading, current_reading, usage")


class WaterBillData(BaseModel):
    """Structured data for water utility bills."""
    # Dates
    billing_period_start: Optional[str] = Field(description="Start date of billing period (YYYY-MM-DD)")
    billing_period_end: Optional[str] = Field(description="End date of billing period (YYYY-MM-DD)")
    statement_date: Optional[str] = Field(description="Statement/bill date (YYYY-MM-DD)")
    due_date: Optional[str] = Field(description="Payment due date (YYYY-MM-DD)")
    
    # Account and facility info
    account_number: Optional[str] = Field(description="Water account number")
    service_address: Optional[str] = Field(description="Service address for water")
    facility_name: Optional[str] = Field(description="Name of the facility being billed")
    facility_address: Optional[str] = Field(description="Full address of the facility")
    
    # Customer info (billed to)
    customer_name: Optional[str] = Field(description="Customer name (company being billed)")
    customer_address: Optional[str] = Field(description="Customer billing address")
    customer_attention: Optional[str] = Field(description="Attention line (e.g., Accounts Payable)")
    
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


class PermitData(BaseModel):
    """Structured data for environmental permits."""
    permit_number: Optional[str] = Field(description="Permit identification number")
    permit_type: Optional[str] = Field(description="Type of permit (air, water, waste, etc.)")
    issue_date: Optional[str] = Field(description="Date permit was issued (YYYY-MM-DD)")
    expiry_date: Optional[str] = Field(description="Date permit expires (YYYY-MM-DD)")
    issuing_authority: Optional[str] = Field(description="Agency that issued the permit")
    
    # Facility info
    facility_name: Optional[str] = Field(description="Name of the permitted facility")
    facility_address: Optional[str] = Field(description="Address of the permitted facility")
    
    # Compliance requirements
    emission_limits: Optional[List[Dict[str, Any]]] = Field(description="Emission or discharge limits")
    monitoring_requirements: Optional[List[str]] = Field(description="Required monitoring activities")
    reporting_frequency: Optional[str] = Field(description="How often reports must be submitted")
    special_conditions: Optional[List[str]] = Field(description="Special permit conditions")
    
    # Permitted activities
    permitted_activities: Optional[List[str]] = Field(description="Activities allowed under permit")
    prohibited_activities: Optional[List[str]] = Field(description="Activities not allowed")


class InvoiceData(BaseModel):
    """Structured data for supplier invoices."""
    invoice_number: Optional[str] = Field(description="Invoice number")
    invoice_date: Optional[str] = Field(description="Invoice date (YYYY-MM-DD)")
    due_date: Optional[str] = Field(description="Payment due date (YYYY-MM-DD)")
    
    # Vendor info
    vendor_name: Optional[str] = Field(description="Vendor/supplier name")
    vendor_address: Optional[str] = Field(description="Vendor address")
    
    # Line items
    line_items: Optional[List[Dict[str, Any]]] = Field(
        description="List of line items with description, quantity, unit price, total"
    )
    
    # Environmental specific
    waste_disposal_items: Optional[List[Dict[str, Any]]] = Field(
        description="Waste disposal related line items"
    )
    recycling_items: Optional[List[Dict[str, Any]]] = Field(
        description="Recycling related line items"
    )
    environmental_fees: Optional[float] = Field(description="Environmental or regulatory fees")
    
    # Totals
    subtotal: Optional[float] = Field(description="Subtotal before tax")
    tax_amount: Optional[float] = Field(description="Tax amount")
    total_amount: Optional[float] = Field(description="Total invoice amount")


class BaseExtractor:
    """Base class for EHS document extractors."""
    
    def __init__(self, llm_model: str = "gpt-4"):
        """
        Initialize the extractor.
        
        Args:
            llm_model: LLM model to use for extraction
        """
        if "claude" in llm_model.lower():
            self.llm = ChatAnthropic(model=llm_model, temperature=0)
        else:
            self.llm = ChatOpenAI(model=llm_model, temperature=0)
    
    def clean_json_string(self, json_str: str) -> str:
        """
        Clean JSON string for parsing.
        
        Args:
            json_str: Raw JSON string
            
        Returns:
            Cleaned JSON string
        """
        # Remove markdown code blocks
        json_str = re.sub(r'^```json\s*', '', json_str, flags=re.MULTILINE)
        json_str = re.sub(r'^```\s*', '', json_str, flags=re.MULTILINE)
        json_str = re.sub(r'\s*```$', '', json_str, flags=re.MULTILINE)
        
        # Remove any trailing commas
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)
        
        return json_str.strip()


class UtilityBillExtractor(BaseExtractor):
    """Extract data from utility bills."""
    
    def __init__(self, llm_model: str = "gpt-4"):
        super().__init__(llm_model)
        
        # Create output parser
        self.parser = JsonOutputParser(pydantic_object=UtilityBillData)
        
        # Create prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at extracting structured data from utility bills.
            Extract all relevant information and return it as JSON.
            Pay special attention to:
            - All dates (billing period, statement date, due date) - convert to YYYY-MM-DD format
            - Distinguish between CUSTOMER (billed to) and SERVICE LOCATION (facility)
            - Customer information (name, billing address, attention line)
            - Facility/Service location (name and address where service is provided)
            - Utility provider details (company name, address, phone, website, payment address)
            - Account number
            - Energy consumption broken down by peak/off-peak if available
            - All meter readings with meter IDs and service types
            - Total costs and detailed charge breakdown
            
            For meter_readings, extract as a list with objects containing:
            - meter_id: The meter identifier (e.g., "MTR-7743-A")
            - type: The meter type (e.g., "electricity", "gas", "water")
            - service_type: The specific service type (e.g., "Commercial - Peak", "Commercial - Off-Peak")
            - previous_reading: Previous meter reading value (number only)
            - current_reading: Current meter reading value (number only)
            - usage: The difference (current - previous)
            
            For charge_breakdown, extract as a list with objects containing:
            - description: Charge description
            - rate: Rate per unit if applicable
            - amount: Dollar amount
            
            For dates, convert to YYYY-MM-DD format.
            For missing values, use null.
            
            {format_instructions}"""),
            ("human", "Extract data from this utility bill:\n\n{content}")
        ])
    
    def extract(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Extract structured data from utility bill content.
        
        Args:
            content: Text content of the utility bill
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
            
            # Log the prompt as user interaction with robust error handling
            try:
                # Use the actual content being extracted rather than formatted_prompt structure
                prompt_content = content  # Use the actual content being extracted
                log_llm_interaction(
                    role="user",
                    content=prompt_content,
                    context={
                        "document_type": "utility_bill",
                        "extractor_name": "UtilityBillExtractor",
                        "timestamp": datetime.now().isoformat(),
                        "metadata": metadata
                    }
                )
                # Forward transcript entry to web app backend
                forward_transcript_entry(
                    role="user",
                    content=prompt_content,
                    context={
                        "document_type": "utility_bill",
                        "extractor_name": "UtilityBillExtractor",
                        "timestamp": datetime.now().isoformat(),
                        "metadata": metadata
                    }
                )
            except Exception as log_error:
                logger.warning(f"Failed to log user prompt interaction: {log_error}")
            
            # Get LLM response
            response = self.llm.invoke(formatted_prompt)
            
            # Log the LLM response as assistant interaction with robust error handling
            try:
                log_llm_interaction(
                    role="assistant",
                    content=response.content,
                    context={
                        "document_type": "utility_bill",
                        "extractor_name": "UtilityBillExtractor",
                        "timestamp": datetime.now().isoformat(),
                        "metadata": metadata
                    }
                )
                # Forward transcript entry to web app backend
                forward_transcript_entry(
                    role="assistant",
                    content=response.content,
                    context={
                        "document_type": "utility_bill",
                        "extractor_name": "UtilityBillExtractor",
                        "timestamp": datetime.now().isoformat(),
                        "metadata": metadata
                    }
                )
            except Exception as log_error:
                logger.warning(f"Failed to log assistant response interaction: {log_error}")
            
            # Parse response
            extracted_data = self.parser.parse(response.content)
            
            # Add metadata
            if metadata:
                extracted_data["metadata"] = metadata
            
            logger.info(f"Successfully extracted utility bill data")
            return extracted_data
            
        except Exception as e:
            logger.error(f"Error extracting utility bill data: {str(e)}")
            # Return empty structure on error
            return UtilityBillData().dict()


class WaterBillExtractor(BaseExtractor):
    """Extract data from water bills."""
    
    def __init__(self, llm_model: str = "gpt-4"):
        super().__init__(llm_model)
        
        # Create output parser
        self.parser = JsonOutputParser(pydantic_object=WaterBillData)
        
        # Create prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at extracting structured data from water utility bills.
            Extract all relevant information and return it as JSON.
            Pay special attention to:
            - All dates (billing period, statement date, due date) - convert to YYYY-MM-DD format
            - Distinguish between CUSTOMER (billed to) and SERVICE LOCATION (facility)
            - Customer information (name, billing address, attention line)
            - Facility/Service location (name and address where water service is provided)
            - Water utility provider details (company name, address, phone, website, payment address)
            - Account number
            - Water consumption in gallons (and cubic meters if provided)
            - All charges including sewer service, stormwater fees, conservation taxes
            - Meter readings with units (gallons, CCF, cubic meters)
            
            For meter_readings, extract as a list with objects containing:
            - meter_id: The meter identifier
            - type: "water"
            - service_type: The specific service type (e.g., "Domestic Water", "Irrigation")
            - previous_reading: Previous meter reading value (number only)
            - current_reading: Current meter reading value (number only)
            - usage: The difference (current - previous)
            - unit: The unit of measurement (gallons, CCF, cubic meters, etc.)
            
            Note: Water providers are often municipal/government entities.
            
            For dates, convert to YYYY-MM-DD format.
            For missing values, use null.
            
            {format_instructions}"""),
            ("human", "Extract data from this water bill:\n\n{content}")
        ])
    
    def extract(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Extract structured data from water bill content.
        
        Args:
            content: Text content of the water bill
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
            
            # Log the prompt as user interaction with robust error handling
            try:
                # Use the actual content being extracted rather than formatted_prompt structure
                prompt_content = content  # Use the actual content being extracted
                log_llm_interaction(
                    role="user",
                    content=prompt_content,
                    context={
                        "document_type": "water_bill",
                        "extractor_name": "WaterBillExtractor",
                        "timestamp": datetime.now().isoformat(),
                        "metadata": metadata
                    }
                )
                # Forward transcript entry to web app backend
                forward_transcript_entry(
                    role="user",
                    content=prompt_content,
                    context={
                        "document_type": "water_bill",
                        "extractor_name": "WaterBillExtractor",
                        "timestamp": datetime.now().isoformat(),
                        "metadata": metadata
                    }
                )
            except Exception as log_error:
                logger.warning(f"Failed to log user prompt interaction: {log_error}")
            
            # Get LLM response
            response = self.llm.invoke(formatted_prompt)
            
            # Log the LLM response as assistant interaction with robust error handling
            try:
                log_llm_interaction(
                    role="assistant",
                    content=response.content,
                    context={
                        "document_type": "water_bill",
                        "extractor_name": "WaterBillExtractor",
                        "timestamp": datetime.now().isoformat(),
                        "metadata": metadata
                    }
                )
                # Forward transcript entry to web app backend
                forward_transcript_entry(
                    role="assistant",
                    content=response.content,
                    context={
                        "document_type": "water_bill",
                        "extractor_name": "WaterBillExtractor",
                        "timestamp": datetime.now().isoformat(),
                        "metadata": metadata
                    }
                )
            except Exception as log_error:
                logger.warning(f"Failed to log assistant response interaction: {log_error}")
            
            # Parse response
            extracted_data = self.parser.parse(response.content)
            
            # Add metadata
            if metadata:
                extracted_data["metadata"] = metadata
            
            logger.info(f"Successfully extracted water bill data")
            return extracted_data
            
        except Exception as e:
            logger.error(f"Error extracting water bill data: {str(e)}")
            # Return empty structure on error
            return WaterBillData().dict()


class PermitExtractor(BaseExtractor):
    """Extract data from environmental permits."""
    
    def __init__(self, llm_model: str = "gpt-4"):
        super().__init__(llm_model)
        
        # Create output parser
        self.parser = JsonOutputParser(pydantic_object=PermitData)
        
        # Create prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at extracting structured data from environmental permits.
            Extract all relevant information and return it as JSON.
            Pay special attention to:
            - Permit number and type
            - Issue and expiry dates
            - Facility information
            - Emission/discharge limits
            - Monitoring and reporting requirements
            - Special conditions
            
            For dates, use YYYY-MM-DD format.
            For missing values, use null.
            
            {format_instructions}"""),
            ("human", "Extract data from this permit document:\n\n{content}")
        ])
    
    def extract(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Extract structured data from permit content.
        
        Args:
            content: Text content of the permit
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
            
            # Log the prompt as user interaction with robust error handling
            try:
                # Use the actual content being extracted rather than formatted_prompt structure
                prompt_content = content  # Use the actual content being extracted
                log_llm_interaction(
                    role="user",
                    content=prompt_content,
                    context={
                        "document_type": "permit",
                        "extractor_name": "PermitExtractor",
                        "timestamp": datetime.now().isoformat(),
                        "metadata": metadata
                    }
                )
                # Forward transcript entry to web app backend
                forward_transcript_entry(
                    role="user",
                    content=prompt_content,
                    context={
                        "document_type": "permit",
                        "extractor_name": "PermitExtractor",
                        "timestamp": datetime.now().isoformat(),
                        "metadata": metadata
                    }
                )
            except Exception as log_error:
                logger.warning(f"Failed to log user prompt interaction: {log_error}")
            
            # Get LLM response
            response = self.llm.invoke(formatted_prompt)
            
            # Log the LLM response as assistant interaction with robust error handling
            try:
                log_llm_interaction(
                    role="assistant",
                    content=response.content,
                    context={
                        "document_type": "permit",
                        "extractor_name": "PermitExtractor",
                        "timestamp": datetime.now().isoformat(),
                        "metadata": metadata
                    }
                )
                # Forward transcript entry to web app backend
                forward_transcript_entry(
                    role="assistant",
                    content=response.content,
                    context={
                        "document_type": "permit",
                        "extractor_name": "PermitExtractor",
                        "timestamp": datetime.now().isoformat(),
                        "metadata": metadata
                    }
                )
            except Exception as log_error:
                logger.warning(f"Failed to log assistant response interaction: {log_error}")
            
            # Parse response
            extracted_data = self.parser.parse(response.content)
            
            # Add metadata
            if metadata:
                extracted_data["metadata"] = metadata
            
            logger.info(f"Successfully extracted permit data")
            return extracted_data
            
        except Exception as e:
            logger.error(f"Error extracting permit data: {str(e)}")
            return PermitData().dict()


class InvoiceExtractor(BaseExtractor):
    """Extract data from supplier invoices."""
    
    def __init__(self, llm_model: str = "gpt-4"):
        super().__init__(llm_model)
        
        # Create output parser
        self.parser = JsonOutputParser(pydantic_object=InvoiceData)
        
        # Create prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at extracting structured data from invoices, 
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
            
            {format_instructions}"""),
            ("human", "Extract data from this invoice:\n\n{content}")
        ])
    
    def extract(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Extract structured data from invoice content.
        
        Args:
            content: Text content of the invoice
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
            
            # Log the prompt as user interaction with robust error handling
            try:
                # Use the actual content being extracted rather than formatted_prompt structure
                prompt_content = content  # Use the actual content being extracted
                log_llm_interaction(
                    role="user",
                    content=prompt_content,
                    context={
                        "document_type": "invoice",
                        "extractor_name": "InvoiceExtractor",
                        "timestamp": datetime.now().isoformat(),
                        "metadata": metadata
                    }
                )
                # Forward transcript entry to web app backend
                forward_transcript_entry(
                    role="user",
                    content=prompt_content,
                    context={
                        "document_type": "invoice",
                        "extractor_name": "InvoiceExtractor",
                        "timestamp": datetime.now().isoformat(),
                        "metadata": metadata
                    }
                )
            except Exception as log_error:
                logger.warning(f"Failed to log user prompt interaction: {log_error}")
            
            # Get LLM response
            response = self.llm.invoke(formatted_prompt)
            
            # Log the LLM response as assistant interaction with robust error handling
            try:
                log_llm_interaction(
                    role="assistant",
                    content=response.content,
                    context={
                        "document_type": "invoice",
                        "extractor_name": "InvoiceExtractor",
                        "timestamp": datetime.now().isoformat(),
                        "metadata": metadata
                    }
                )
                # Forward transcript entry to web app backend
                forward_transcript_entry(
                    role="assistant",
                    content=response.content,
                    context={
                        "document_type": "invoice",
                        "extractor_name": "InvoiceExtractor",
                        "timestamp": datetime.now().isoformat(),
                        "metadata": metadata
                    }
                )
            except Exception as log_error:
                logger.warning(f"Failed to log assistant response interaction: {log_error}")
            
            # Parse response
            extracted_data = self.parser.parse(response.content)
            
            # Add metadata
            if metadata:
                extracted_data["metadata"] = metadata
            
            logger.info(f"Successfully extracted invoice data")
            return extracted_data
            
        except Exception as e:
            logger.error(f"Error extracting invoice data: {str(e)}")
            return InvoiceData().dict()


class WasteManifestData(BaseModel):
    """Structured data for waste manifests."""
    # Manifest Information
    manifest_tracking_number: Optional[str] = Field(description="Unique manifest tracking number")
    manifest_type: Optional[str] = Field(description="Type of waste manifest (e.g., NON-HAZARDOUS WASTE MANIFEST, HAZARDOUS WASTE MANIFEST)")
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
        description="List of waste items with description (waste type), container_type, container_quantity (number of containers), weight_per_container (if available), total_weight (if available), classification (hazardous/non-hazardous)"
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
    total_waste_quantity: Optional[float] = Field(description="Total weight of all waste items (may be estimated if not explicitly provided)")
    total_waste_unit: Optional[str] = Field(description="Unit for total waste quantity (tons, pounds, etc.)")


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
            
            MANIFEST TYPE AND CLASSIFICATION:
            - Extract manifest_type from the document title/header (e.g., "NON-HAZARDOUS WASTE MANIFEST", "HAZARDOUS WASTE MANIFEST")
            - This determines the waste classification for all items in the manifest
            
            WASTE ITEMS EXTRACTION:
            For waste_items, extract as a list with objects containing:
            - description: Use the waste description field as the waste type (e.g., "Construction Debris", "Metal Scraps")
            - container_type: Type of container used (e.g., "Roll-off Container", "Drum", "Box")
            - container_quantity: The "Quantity" field refers to the NUMBER OF CONTAINERS, not weight
            - weight_per_container: Weight per individual container if explicitly provided
            - total_weight: Total weight for this waste item if explicitly provided
            - classification: Set based on manifest_type ("hazardous" or "non-hazardous")
            - estimated_weight_note: If no explicit weight is given, add note about weight estimation needs
            
            WEIGHT ESTIMATION GUIDANCE:
            When explicit weights are not provided, consider typical container capacities:
            - Roll-off containers: typically hold 10-20 tons depending on size
            - Standard drums: typically 400-500 lbs when full
            - Small containers/boxes: 50-200 lbs depending on material
            Note: Actual weights depend heavily on waste density and should be estimated based on container type and contents.
            
            TOTAL CALCULATIONS:
            - total_waste_quantity: Sum all waste weights if provided, or note that estimation is needed
            - total_waste_unit: Use "tons" for large quantities, "pounds" for smaller amounts
            
            OTHER REQUIRED FIELDS:
            - Manifest tracking number and document details
            - All parties involved (generator, transporter, facility) with EPA IDs
            - All certification dates and signatures
            - Special handling instructions
            
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
            
            # Log the prompt as user interaction with robust error handling
            try:
                # Use the actual content being extracted rather than formatted_prompt structure
                prompt_content = content  # Use the actual content being extracted
                log_llm_interaction(
                    role="user",
                    content=prompt_content,
                    context={
                        "document_type": "waste_manifest",
                        "extractor_name": "WasteManifestExtractor",
                        "timestamp": datetime.now().isoformat(),
                        "metadata": metadata
                    }
                )
                # Forward transcript entry to web app backend
                forward_transcript_entry(
                    role="user",
                    content=prompt_content,
                    context={
                        "document_type": "waste_manifest",
                        "extractor_name": "WasteManifestExtractor",
                        "timestamp": datetime.now().isoformat(),
                        "metadata": metadata
                    }
                )
            except Exception as log_error:
                logger.warning(f"Failed to log user prompt interaction: {log_error}")
            
            # Get LLM response
            response = self.llm.invoke(formatted_prompt)
            
            # Log the LLM response as assistant interaction with robust error handling
            try:
                log_llm_interaction(
                    role="assistant",
                    content=response.content,
                    context={
                        "document_type": "waste_manifest",
                        "extractor_name": "WasteManifestExtractor",
                        "timestamp": datetime.now().isoformat(),
                        "metadata": metadata
                    }
                )
                # Forward transcript entry to web app backend
                forward_transcript_entry(
                    role="assistant",
                    content=response.content,
                    context={
                        "document_type": "waste_manifest",
                        "extractor_name": "WasteManifestExtractor",
                        "timestamp": datetime.now().isoformat(),
                        "metadata": metadata
                    }
                )
            except Exception as log_error:
                logger.warning(f"Failed to log assistant response interaction: {log_error}")
            
            # Parse response
            extracted_data = self.parser.parse(response.content)
            
            # Post-process the data to ensure consistency
            extracted_data = self._post_process_waste_manifest(extracted_data)
            
            # Add metadata
            if metadata:
                extracted_data["metadata"] = metadata
            
            logger.info(f"Successfully extracted waste manifest data")
            return extracted_data
            
        except Exception as e:
            logger.error(f"Error extracting waste manifest data: {str(e)}")
            # Return empty structure on error
            return WasteManifestData().dict()
    
    def _post_process_waste_manifest(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Post-process extracted waste manifest data for consistency.
        Estimates weights when not provided based on container types.
        
        Args:
            data: Raw extracted data
            
        Returns:
            Processed data with consistent formatting and weight estimates
        """
        # Ensure waste classification consistency
        if data.get("manifest_type") and data.get("waste_items"):
            # Determine classification from manifest type
            classification = "non-hazardous"
            if data["manifest_type"] and "hazardous" in data["manifest_type"].lower() and "non-hazardous" not in data["manifest_type"].lower():
                classification = "hazardous"
            
            # Apply classification to all waste items
            for item in data["waste_items"]:
                if not item.get("classification"):
                    item["classification"] = classification
        
        # Process each waste item for weight estimation
        if data.get("waste_items"):
            total_weight = 0
            
            for item in data["waste_items"]:
                container_type = item.get("container_type", "").lower()
                container_quantity = item.get("container_quantity", 0)
                
                # Check if item already has weight information
                if not item.get("total_weight") and container_type and container_quantity:
                    # Estimate weight per container based on container type
                    weight_per_container = 0
                    
                    if "open top roll-off" in container_type or "roll-off" in container_type:
                        weight_per_container = 15  # tons per container
                    elif "drum" in container_type:
                        weight_per_container = 0.25  # tons per container (500 lbs)
                    else:
                        weight_per_container = 1  # tons per container (default)
                    
                    # Set estimated weights
                    item["weight_per_container"] = weight_per_container
                    item["total_weight"] = weight_per_container * container_quantity
                
                # Add to total weight calculation
                item_weight = item.get("total_weight", 0)
                if item_weight:
                    total_weight += item_weight
        
        # Set total waste quantity and unit
        if total_weight > 0:
            data["total_waste_quantity"] = total_weight
            
        # Set total waste unit to "tons" if not provided
        if not data.get("total_waste_unit"):
            data["total_waste_unit"] = "tons"
        
        return data


class EquipmentSpecificationExtractor(BaseExtractor):
    """Extract data from equipment specifications."""
    
    def __init__(self, llm_model: str = "gpt-4"):
        super().__init__(llm_model)
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at extracting environmental and operational data 
            from equipment specifications and datasheets.
            Extract all relevant information and return it as JSON.
            Focus on:
            - Equipment model and manufacturer
            - Energy efficiency ratings
            - Operating parameters
            - Emission factors
            - Environmental compliance features
            - Maintenance requirements
            
            Return extracted data as a JSON object."""),
            ("human", "Extract data from this equipment specification:\n\n{content}")
        ])
    
    def extract(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Extract structured data from equipment specifications.
        
        Args:
            content: Text content of the specification
            metadata: Optional metadata about the document
            
        Returns:
            Dictionary of extracted data
        """
        try:
            # Format prompt
            formatted_prompt = self.prompt.format_messages(content=content)
            
            # Log the prompt as user interaction with robust error handling
            try:
                # Use the actual content being extracted rather than formatted_prompt structure
                prompt_content = content  # Use the actual content being extracted
                log_llm_interaction(
                    role="user",
                    content=prompt_content,
                    context={
                        "document_type": "equipment_specification",
                        "extractor_name": "EquipmentSpecificationExtractor",
                        "timestamp": datetime.now().isoformat(),
                        "metadata": metadata
                    }
                )
                # Forward transcript entry to web app backend
                forward_transcript_entry(
                    role="user",
                    content=prompt_content,
                    context={
                        "document_type": "equipment_specification",
                        "extractor_name": "EquipmentSpecificationExtractor",
                        "timestamp": datetime.now().isoformat(),
                        "metadata": metadata
                    }
                )
            except Exception as log_error:
                logger.warning(f"Failed to log user prompt interaction: {log_error}")
            
            # Get LLM response
            response = self.llm.invoke(formatted_prompt)
            
            # Log the LLM response as assistant interaction with robust error handling
            try:
                log_llm_interaction(
                    role="assistant",
                    content=response.content,
                    context={
                        "document_type": "equipment_specification",
                        "extractor_name": "EquipmentSpecificationExtractor",
                        "timestamp": datetime.now().isoformat(),
                        "metadata": metadata
                    }
                )
                # Forward transcript entry to web app backend
                forward_transcript_entry(
                    role="assistant",
                    content=response.content,
                    context={
                        "document_type": "equipment_specification",
                        "extractor_name": "EquipmentSpecificationExtractor",
                        "timestamp": datetime.now().isoformat(),
                        "metadata": metadata
                    }
                )
            except Exception as log_error:
                logger.warning(f"Failed to log assistant response interaction: {log_error}")
            
            # Parse JSON from response
            json_str = self.clean_json_string(response.content)
            extracted_data = json.loads(json_str)
            
            # Add metadata
            if metadata:
                extracted_data["metadata"] = metadata
            
            logger.info(f"Successfully extracted equipment specification data")
            return extracted_data
            
        except Exception as e:
            logger.error(f"Error extracting equipment spec data: {str(e)}")
            return {}


# Utility function for quick extraction
def extract_ehs_data(
    content: str,
    document_type: str,
    llm_model: str = "gpt-4",
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Extract EHS data from document content.
    
    Args:
        content: Document text content
        document_type: Type of document (utility_bill, water_bill, permit, invoice, equipment_spec, waste_manifest)
        llm_model: LLM model to use
        metadata: Optional document metadata
        
    Returns:
        Dictionary of extracted data
    """
    extractors = {
        "utility_bill": UtilityBillExtractor,
        "water_bill": WaterBillExtractor,
        "permit": PermitExtractor,
        "invoice": InvoiceExtractor,
        "equipment_spec": EquipmentSpecificationExtractor,
        "waste_manifest": WasteManifestExtractor
    }
    
    extractor_class = extractors.get(document_type, InvoiceExtractor)
    extractor = extractor_class(llm_model=llm_model)
    
    return extractor.extract(content, metadata)