# Executive Dashboard JSON Data Structures

> Created: 2025-08-27
> Version: 1.0.0
> Component: AnalyticsExecutive.js

## Overview

This document describes all JSON data structures required for the Analytics (Executive) dashboard. The dashboard provides executive-level insights into EHS performance across location hierarchies and time periods, featuring a global scorecard and detailed thematic analysis for Electricity, Water, and Waste management.

## Complete Data Structure

### Main Dashboard Response

```json
{
  "globalScorecard": {
    "ehsScore": 87,
    "costSavings": 245000,
    "carbonFootprint": 1250,
    "riskLevel": "medium"
  },
  "electricity": {
    "goal": {
      "actual": 85,
      "target": 80,
      "unit": "% efficiency"
    },
    "facts": {
      "consumption": "2.4M kWh",
      "cost": "$340K",
      "co2": "720 tons CO₂"
    },
    "trend": [78, 82, 79, 85, 87, 85, 88, 85],
    "analysis": "Energy consumption trending up 8% this quarter due to increased production",
    "recommendation": "Install LED lighting and smart HVAC controls to reduce consumption by 15%"
  },
  "water": {
    "goal": {
      "actual": 72,
      "target": 85,
      "unit": "% efficiency"
    },
    "facts": {
      "consumption": "150K gallons",
      "cost": "$45K",
      "co2": "85 tons CO₂"
    },
    "trend": [85, 82, 78, 72, 69, 72, 75, 72],
    "analysis": "Water efficiency below target due to cooling system inefficiencies",
    "recommendation": "Upgrade cooling tower systems and implement water recycling program"
  },
  "waste": {
    "goal": {
      "actual": 92,
      "target": 90,
      "unit": "% diversion"
    },
    "facts": {
      "generation": "45 tons",
      "cost": "$12K saved",
      "co2": "120 tons CO₂ avoided"
    },
    "trend": [88, 90, 89, 92, 94, 92, 91, 92],
    "analysis": "Waste diversion exceeding targets with strong recycling program performance",
    "recommendation": "Expand composting program to achieve 95% diversion rate"
  }
}
```

## Detailed Field Descriptions

### Global Scorecard Structure

The `globalScorecard` object contains high-level EHS performance indicators:

| Field | Type | Description | Example Values |
|-------|------|-------------|----------------|
| `ehsScore` | Integer | Overall EHS performance score (0-100) | 87 |
| `costSavings` | Integer | Total cost savings in USD | 245000 |
| `carbonFootprint` | Integer | Total CO₂ reduction in tons | 1250 |
| `riskLevel` | String | Overall risk assessment | "low", "medium", "high" |

### Thematic Card Structure

Each thematic card (`electricity`, `water`, `waste`) follows the same structure:

#### Goal Performance Object
```json
{
  "actual": 85,
  "target": 80,
  "unit": "% efficiency"
}
```

| Field | Type | Description | Notes |
|-------|------|-------------|-------|
| `actual` | Number | Current performance value | Can be integer or decimal |
| `target` | Number | Target performance value | Used for gauge visualization |
| `unit` | String | Unit of measurement | Displayed in UI |

#### Key Facts Object
```json
{
  "consumption": "2.4M kWh",
  "cost": "$340K", 
  "co2": "720 tons CO₂"
}
```

| Field | Type | Description | Variations by Theme |
|-------|------|-------------|---------------------|
| First fact | String | Primary metric | `consumption` (electricity/water), `generation` (waste) |
| `cost` | String | Cost impact | Can be positive cost or savings |
| `co2` | String | Carbon impact | Can be emissions or reductions |

#### Trend Analysis
```json
{
  "trend": [78, 82, 79, 85, 87, 85, 88, 85]
}
```

| Field | Type | Description | Requirements |
|-------|------|-------------|--------------|
| `trend` | Array of Numbers | Historical performance values | Minimum 2 values, typically 8 periods |

#### Analysis and Recommendations
```json
{
  "analysis": "Energy consumption trending up 8% this quarter due to increased production",
  "recommendation": "Install LED lighting and smart HVAC controls to reduce consumption by 15%"
}
```

| Field | Type | Description | Purpose |
|-------|------|-------------|---------|
| `analysis` | String | Current performance analysis | Contextual explanation of trends |
| `recommendation` | String | Actionable improvement suggestion | Executive-level recommendations |

## Filtering Parameters

### Location Hierarchy
The dashboard supports hierarchical location filtering:

```javascript
const locationOptions = [
  { value: "global", label: "Global View" },
  { value: "northamerica", label: "North America" },
  { value: "illinois", label: "Illinois" },
  { value: "algonquin", label: "Algonquin Site" }
];
```

### Date Range Options
The dashboard supports multiple time period selections:

```javascript
const dateRangeOptions = [
  { value: "30days", label: "Last 30 Days" },
  { value: "quarter", label: "This Quarter" },
  { value: "custom", label: "Custom Range" }
];
```

## API Endpoint Requirements

### Endpoint Structure
```
GET /api/executive-dashboard
```

### Query Parameters
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `location` | String | Yes | Location filter (global, northamerica, illinois, algonquin) |
| `dateRange` | String | Yes | Date range filter (30days, quarter, custom) |
| `startDate` | String | Conditional | Required when dateRange=custom (YYYY-MM-DD) |
| `endDate` | String | Conditional | Required when dateRange=custom (YYYY-MM-DD) |

### Example Request
```
GET /api/executive-dashboard?location=algonquin&dateRange=30days
```

## Calculated Fields and Aggregations

### Performance Score Calculations

#### Color Coding Logic
- **Green** (`#4caf50`): `actual >= target`
- **Orange** (`#ff9800`): `actual >= target * 0.8`
- **Red** (`#f44336`): `actual < target * 0.8`

#### Trend Direction Logic
```javascript
// Compare recent 3 periods vs previous 3 periods
const recent = trend.slice(-3).reduce((a, b) => a + b, 0) / 3;
const previous = trend.slice(-6, -3).reduce((a, b) => a + b, 0) / 3;
const isUpward = recent > previous;
```

### Data Aggregation Rules

#### Location-Based Aggregation
- **Global**: Aggregate all locations
- **Regional**: Aggregate all sites within region
- **State**: Aggregate all sites within state
- **Site**: Individual site data

#### Time-Based Aggregation
- **Last 30 Days**: Rolling 30-day period
- **This Quarter**: Current quarter to date
- **Custom Range**: User-specified date range

## UI Rendering Specifications

### Gauge Chart Requirements
- **Range**: 0 to target value
- **Fill**: Based on actual vs target performance
- **Colors**: Follow performance score color coding
- **Display**: Show actual value prominently with "of target" label

### Sparkline Chart Requirements
- **Data Points**: 8 historical values
- **Visualization**: Simple line chart with final point highlighted
- **Scaling**: Auto-scale to data range
- **Colors**: Primary blue (#1976d2)

### Risk Level Display
Risk levels should be displayed as colored chips:
- **Low**: Green chip
- **Medium**: Orange chip  
- **High**: Red chip

## Data Validation Requirements

### Required Fields
All fields in the main structure are required. Missing fields will cause UI rendering errors.

### Data Type Validation
- Numbers must be valid numeric values
- Strings must not be empty
- Arrays must contain at least 2 elements for trend analysis
- Risk level must be one of: "low", "medium", "high"

### Performance Targets
- Goal actual/target values should be positive numbers
- Trend arrays should contain numeric values only
- Cost values should include currency symbols and appropriate formatting
- Environmental impact values should include appropriate units

## Example API Responses

### Successful Response
```json
{
  "status": "success",
  "data": {
    "globalScorecard": {
      "ehsScore": 87,
      "costSavings": 245000,
      "carbonFootprint": 1250,
      "riskLevel": "medium"
    },
    "electricity": {
      "goal": { "actual": 85, "target": 80, "unit": "% efficiency" },
      "facts": {
        "consumption": "2.4M kWh",
        "cost": "$340K",
        "co2": "720 tons CO₂"
      },
      "trend": [78, 82, 79, 85, 87, 85, 88, 85],
      "analysis": "Energy consumption trending up 8% this quarter due to increased production",
      "recommendation": "Install LED lighting and smart HVAC controls to reduce consumption by 15%"
    },
    "water": {
      "goal": { "actual": 72, "target": 85, "unit": "% efficiency" },
      "facts": {
        "consumption": "150K gallons",
        "cost": "$45K",
        "co2": "85 tons CO₂"
      },
      "trend": [85, 82, 78, 72, 69, 72, 75, 72],
      "analysis": "Water efficiency below target due to cooling system inefficiencies",
      "recommendation": "Upgrade cooling tower systems and implement water recycling program"
    },
    "waste": {
      "goal": { "actual": 92, "target": 90, "unit": "% diversion" },
      "facts": {
        "generation": "45 tons",
        "cost": "$12K saved",
        "co2": "120 tons CO₂ avoided"
      },
      "trend": [88, 90, 89, 92, 94, 92, 91, 92],
      "analysis": "Waste diversion exceeding targets with strong recycling program performance",
      "recommendation": "Expand composting program to achieve 95% diversion rate"
    }
  },
  "metadata": {
    "location": "algonquin",
    "dateRange": "30days",
    "generatedAt": "2025-08-27T10:30:00Z",
    "dataVersion": "1.0"
  }
}
```

### Error Response
```json
{
  "status": "error",
  "error": {
    "code": "INVALID_LOCATION",
    "message": "Invalid location parameter provided",
    "details": "Location 'invalid_site' not found in hierarchy"
  }
}
```

## Implementation Notes

### Frontend Integration
- Component expects data in exact structure shown above
- Loading states should be handled during API calls
- Error states should display user-friendly messages
- Data should be cached appropriately to avoid unnecessary API calls

### Backend Considerations  
- Data should be pre-aggregated where possible for performance
- Cache frequently accessed data combinations
- Implement proper error handling for missing data
- Validate location hierarchy permissions
- Consider implementing data freshness indicators

### Performance Optimization
- Trend data should be limited to necessary time periods
- Consider implementing pagination for large datasets
- Use appropriate database indexes for location and date filtering
- Implement caching strategies for executive-level aggregated data

## Future Enhancements

### Planned Extensions
- Add drill-down capability for detailed analysis
- Include comparative benchmarking data
- Add export functionality for reports
- Implement real-time data updates
- Add forecasting and predictive analytics

### Data Structure Extensibility
The current structure allows for easy extension with additional:
- Thematic categories (air quality, safety metrics, etc.)
- Metadata fields for enhanced context
- Multi-currency support for global operations
- Localization support for international deployments