# Risk Assessment Agent Unit Conversion Fixes

## Summary of Changes

Fixed the unit conversion issue in the Risk Assessment Agent to properly handle CO2e emissions for electricity consumption and ensure accurate risk assessment for sites with goals measured in "tonnes CO2e".

## Files Modified

### 1. `/src/agents/risk_assessment_agent.py`

#### Changes Made:

1. **Added CO2e Conversion Factor (Line 70)**
   - Added `ELECTRICITY_CO2E_FACTOR = 0.000395` (tonnes CO2e per kWh - US grid average)
   - This standard emission factor converts electricity consumption to CO2e emissions

2. **Enhanced `get_6month_consumption_data` Method (Lines 437-562)**
   - **Added CO2e calculation for electricity consumption**:
     - Calculates `co2e_emissions = amount * self.ELECTRICITY_CO2E_FACTOR` for each electricity record
     - Adds CO2e data to monthly aggregations with proper units
     - Includes total CO2e emissions in return data
   - **Enhanced monthly data structure**:
     - For electricity category, adds `co2e_emissions` and `co2e_unit` fields
     - Maintains backward compatibility with existing kWh data
   - **Improved data tracking**:
     - Tracks both kWh consumption and CO2e emissions separately
     - Provides totals for both metrics

3. **Updated `llm_compare_to_goals` Method (Lines 210-313)**
   - **Enhanced method signature**: Added `historical_data` parameter to access raw consumption data
   - **Smart unit handling**:
     - Detects when goal unit is "tonnes CO2e" and category is "electricity" 
     - Uses CO2e values for analysis when goal requires it
     - Falls back to raw consumption values for other units
   - **Improved baseline calculation**:
     - Calculates baseline consumption from first month of data
     - Handles CO2e conversion for baseline when needed
     - Passes both actual and baseline consumption to LLM
   - **Enhanced goal_details structure**:
     - Added `baseline_consumption` field for proper gap analysis
     - Provides actual consumption values instead of just change rates
     - Maintains proper unit consistency

4. **Updated `analyze_site_performance` Method (Line 129)**
   - Modified to pass `historical_data` to `llm_compare_to_goals` method
   - Ensures proper data flow for unit conversion logic

5. **Enhanced `get_annual_reduction_goal` Method (Lines 564-595)**
   - Added `category` to return data for better context in analysis
   - Maintains compatibility with existing goal structure

### 2. `/src/agents/prompts/risk_assessment_prompts.py`

#### Changes Made:

1. **Updated Risk Assessment Prompt (Lines 25-51)**
   - **Added baseline consumption context**: Includes baseline consumption in prompt
   - **Enhanced goal interpretation**: Added guidance for percentage-based reduction goals
   - **Improved CO2e handling**: Specific instructions for CO2e targets and reduction calculations
   - **Better risk calculation**: Considers baseline vs current performance for more accurate assessment

2. **Updated `format_risk_assessment_prompt` Function (Lines 102-129)**
   - **Enhanced parameter handling**: Added `baseline_consumption` field with safe fallback
   - **Improved docstring**: Updated documentation to reflect new baseline parameter
   - **Maintained backward compatibility**: Uses `.get()` for safe access to new fields

## Technical Benefits

### 1. Accurate Unit Conversion
- Electricity consumption properly converted to CO2e using standard emission factors
- Automatic unit detection and conversion based on goal requirements
- Maintains data integrity across different measurement units

### 2. Improved Risk Assessment
- LLM receives actual consumption values instead of just change rates
- Baseline consumption enables proper gap analysis
- CO2e emissions properly compared against CO2e goals

### 3. Enhanced Data Quality
- Monthly data includes both raw consumption and CO2e emissions
- Proper unit labeling for all metrics
- Comprehensive totals for analysis

### 4. Better Goal Comparison
- Smart detection of goal units for appropriate data selection
- Proper handling of percentage-based reduction goals
- Accurate projection calculations

## Expected Impact

### For Algonquin IL Site:
- Risk assessment will now properly compare actual CO2e emissions against the CO2e reduction goal
- Should generate HIGH/CRITICAL risk levels when CO2e emissions exceed targets
- LLM will receive accurate baseline and current emissions data for proper analysis

### General Improvements:
- More accurate risk assessments across all sites with CO2e goals
- Better trend analysis with proper unit conversions
- Improved recommendations based on accurate consumption data

## Testing Recommendations

1. **Test with Algonquin IL**: Verify that HIGH/CRITICAL risk is generated for electricity/CO2e goals
2. **Test unit consistency**: Ensure kWh goals still work with raw consumption data
3. **Test baseline calculations**: Verify baseline consumption is properly calculated and used
4. **Test LLM prompts**: Ensure enhanced prompts provide better analysis quality

## Backward Compatibility

All changes maintain backward compatibility:
- Existing kWh-based analysis continues to work
- New CO2e functionality only activates when needed
- All existing API contracts preserved
- Safe fallbacks for missing data fields