import React, { useState, useEffect } from 'react';
import {
  Card,
  CardContent,
  Grid,
  Typography,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Box,
  Chip,
  LinearProgress,
  Button,
  ButtonGroup,
  Alert,
  ListSubheader
} from '@mui/material';
import {
  TrendingUp,
  TrendingDown,
  Water,
  Bolt,
  Delete,
  Park,
  MonetizationOn,
  Warning,
  CheckCircle,
  Info,
  LightbulbOutlined
} from '@mui/icons-material';
import { API_ENDPOINTS } from '../config/api';

const AnalyticsExecutive = () => {
  const [selectedLocation, setSelectedLocation] = useState('algonquin_il');
  const [selectedDateRange, setSelectedDateRange] = useState('30days');
  const [electricityData, setElectricityData] = useState(null);
  const [waterData, setWaterData] = useState(null);
  const [wasteData, setWasteData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Location hierarchy mapping for display
  const locationHierarchy = {
    'algonquin_il': 'Global View → North America → Illinois → Algonquin Site',
    'houston_tx': 'Global View → North America → Texas → Houston Site'
  };

  // Site ID mapping
  const siteIdMapping = {
    'algonquin_il': 'algonquin_il',
    'houston_tx': 'houston_tx'
  };

  // Fetch data from environmental dashboard APIs
  const fetchEnvironmentalData = async () => {
    setLoading(true);
    setError(null);
    setElectricityData(null);
    setWaterData(null);
    setWasteData(null);
    
    try {
      const siteId = siteIdMapping[selectedLocation];
      
      // Make three parallel API calls
      const [electricityResponse, waterResponse, wasteResponse] = await Promise.all([
        fetch(`${API_ENDPOINTS.dashboardElectricity}?site_id=${siteId}&date_range=${selectedDateRange}`),
        fetch(`${API_ENDPOINTS.dashboardWater}?site_id=${siteId}&date_range=${selectedDateRange}`),
        fetch(`${API_ENDPOINTS.dashboardWaste}?site_id=${siteId}&date_range=${selectedDateRange}`)
      ]);

      // Check if all responses are ok
      if (!electricityResponse.ok) {
        throw new Error(`Electricity API error: ${electricityResponse.status}`);
      }
      if (!waterResponse.ok) {
        throw new Error(`Water API error: ${waterResponse.status}`);
      }
      if (!wasteResponse.ok) {
        throw new Error(`Waste API error: ${wasteResponse.status}`);
      }

      // Parse JSON responses
      const [electricity, water, waste] = await Promise.all([
        electricityResponse.json(),
        waterResponse.json(),
        wasteResponse.json()
      ]);

      setElectricityData(electricity);
      setWaterData(water);
      setWasteData(waste);
    } catch (err) {
      console.error('Error fetching environmental data:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchEnvironmentalData();
  }, [selectedLocation, selectedDateRange]);

  const getScoreColor = (score, target) => {
    if (score >= target) return '#4caf50';
    if (score >= target * 0.8) return '#ff9800';
    return '#f44336';
  };

  const getTrendIcon = (trend) => {
    if (typeof trend === 'string') {
      switch (trend.toLowerCase()) {
        case 'improvement':
        case 'improving':
        case 'up':
        case 'positive':
          return <TrendingUp sx={{ color: '#4caf50' }} />;
        case 'concern':
        case 'declining':
        case 'down':
        case 'negative':
          return <TrendingDown sx={{ color: '#f44336' }} />;
        case 'mixed':
        case 'stable':
        case 'flat':
        default:
          return <Info />;
      }
    }

    if (!Array.isArray(trend) || trend.length < 2) return <Info />;
    
    const recent = trend.slice(-3).reduce((a, b) => a + b, 0) / 3;
    const previous = trend.slice(-6, -3).reduce((a, b) => a + b, 0) / 3;
    
    if (recent > previous) return <TrendingUp sx={{ color: '#4caf50' }} />;
    return <TrendingDown sx={{ color: '#f44336' }} />;
  };

  const renderSparkline = (data) => {
    if (!Array.isArray(data) || data.length === 0) {
      return null;
    }

    const max = Math.max(...data);
    const min = Math.min(...data);
    const range = max - min || 1;
    
    const points = data.map((value, index) => {
      const x = (index / (data.length - 1)) * 100;
      const y = 100 - ((value - min) / range) * 100;
      return `${x},${y}`;
    }).join(' ');

    return (
      <svg width="100%" height="40" viewBox="0 0 100 100" style={{ overflow: 'visible' }}>
        <polyline
          points={points}
          fill="none"
          stroke="#1976d2"
          strokeWidth="2"
          vectorEffect="non-scaling-stroke"
        />
        <circle
          cx={data.length > 0 ? ((data.length - 1) / (data.length - 1)) * 100 : 0}
          cy={data.length > 0 ? 100 - ((data[data.length - 1] - min) / range) * 100 : 0}
          r="2"
          fill="#1976d2"
        />
      </svg>
    );
  };

  const renderGauge = (actual, target, unit) => {
    const percentage = (actual / target) * 100;
    const color = getScoreColor(actual, target);
    
    return (
      <Box sx={{ position: 'relative', width: 120, height: 120, mx: 'auto' }}>
        <svg width="120" height="120" style={{ transform: 'rotate(-90deg)' }}>
          <circle
            cx="60"
            cy="60"
            r="50"
            fill="none"
            stroke="#e0e0e0"
            strokeWidth="8"
          />
          <circle
            cx="60"
            cy="60"
            r="50"
            fill="none"
            stroke={color}
            strokeWidth="8"
            strokeLinecap="round"
            strokeDasharray={`${(actual / target) * 314.16} 314.16`}
            strokeDashoffset="0"
          />
        </svg>
        <Box sx={{
          position: 'absolute',
          top: '50%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
          textAlign: 'center'
        }}>
          <Typography variant="h6" sx={{ fontWeight: 'bold', color }}>
            {actual}
          </Typography>
          <Typography variant="caption" color="textSecondary">
            of {target}
          </Typography>
        </Box>
      </Box>
    );
  };

  const calculateCO2Emissions = (data) => {
    if (!data || !data.facts) return 0;
    
    // Extract consumption values and calculate estimated CO2
    const consumption = Object.values(data.facts).reduce((sum, value) => {
      if (typeof value === 'number') return sum + value;
      const num = parseFloat(value);
      return !isNaN(num) ? sum + num : sum;
    }, 0);
    
    // Simple CO2 calculation (this would be more sophisticated in real implementation)
    return Math.round(consumption * 0.0005); // Approximate CO2 factor
  };

  // Helper function to parse recommendations that might be JSON-like strings
  const parseRecommendations = (recommendations) => {
    if (!Array.isArray(recommendations)) return [];
    
    return recommendations.map(rec => {
      if (typeof rec === 'string') {
        try {
          // Try to parse if it looks like JSON
          if (rec.trim().startsWith('{') || rec.trim().startsWith('[')) {
            const parsed = JSON.parse(rec);
            return parsed.text || parsed.description || rec;
          }
        } catch (e) {
          // If parsing fails, return original string
        }
      }
      return rec;
    });
  };

  const renderEnvironmentalCard = (data, title, icon, color, type) => {
    if (!data) return null;

    const co2Emissions = calculateCO2Emissions(data);
    const goals = data.goals || [];
    const facts = data.facts || {};
    const risks = data.risks || [];
    const rawRecommendations = data.recommendations || [];
    const recommendations = parseRecommendations(rawRecommendations);

    return (
      <Card sx={{ height: '100%', boxShadow: 3 }}>
        <CardContent sx={{ p: 3 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
            {icon}
            <Typography variant="h6" sx={{ fontWeight: 'bold' }}>
              {title}
            </Typography>
          </Box>
          
          {/* Goals Section */}
          {goals.length > 0 && (
            <Box sx={{ mb: 3, p: 2, backgroundColor: '#f8f9fa', borderRadius: 1 }}>
              <Typography variant="subtitle1" sx={{ mb: 2, fontWeight: 'bold' }}>
                Goals
              </Typography>
              {goals.slice(0, 2).map((goal, index) => (
                <Typography key={index} variant="body2" sx={{ mb: 1 }}>
                  • {goal}
                </Typography>
              ))}
            </Box>
          )}

          {/* Facts Section - 6-month historical data */}
          <Box sx={{ mb: 3, p: 2, backgroundColor: '#e3f2fd', borderRadius: 1 }}>
            <Typography variant="subtitle1" sx={{ mb: 2, fontWeight: 'bold' }}>
              6-Month Historical Facts
            </Typography>
            {Object.entries(facts).slice(0, 3).map(([key, value]) => (
              <Typography key={key} variant="body2" sx={{ mb: 1 }}>
                <strong>{key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}:</strong> {value}
              </Typography>
            ))}
            {(type === 'electricity' || type === 'water') && (
              <Typography variant="body2" sx={{ mb: 1 }}>
                <strong>Est. CO₂ Emissions:</strong> {co2Emissions} tons
              </Typography>
            )}
          </Box>

          {/* Risk Levels Section */}
          {risks.length > 0 && (
            <Box sx={{ mb: 3, p: 2, backgroundColor: '#fff3e0', borderRadius: 1 }}>
              <Typography variant="subtitle1" sx={{ mb: 2, fontWeight: 'bold' }}>
                Risk Assessment
              </Typography>
              {risks.slice(0, 2).map((risk, index) => {
                // Handle both object and string risk formats
                let riskSeverity = '';
                let riskDescription = '';
                
                if (typeof risk === 'object' && risk !== null) {
                  riskSeverity = risk.severity || '';
                  riskDescription = risk.description || '';
                } else if (typeof risk === 'string') {
                  // If it's a string, use it as description and try to extract severity
                  riskDescription = risk;
                  riskSeverity = risk.toLowerCase().includes('high') ? 'HIGH' :
                               risk.toLowerCase().includes('medium') ? 'MEDIUM' :
                               risk.toLowerCase().includes('low') ? 'LOW' : 'MEDIUM';
                }

                const chipColor = riskSeverity.toLowerCase() === 'low' ? 'success' :
                                riskSeverity.toLowerCase() === 'medium' ? 'warning' : 'error';

                return (
                  <Box key={index} sx={{ mb: 1 }}>
                    <Chip 
                      label={`${riskSeverity} Risk`}
                      color={chipColor}
                      variant="outlined"
                      size="small"
                      sx={{ mr: 1 }}
                    />
                    {riskDescription && (
                      <Typography variant="body2" sx={{ mt: 0.5, color: 'text.secondary' }}>
                        {riskDescription}
                      </Typography>
                    )}
                  </Box>
                );
              })}
            </Box>
          )}

          {/* Recommendations Section */}
          {recommendations.length > 0 && (
            <Box sx={{ p: 2, backgroundColor: '#e8f5e8', borderRadius: 1 }}>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <LightbulbOutlined sx={{ color: '#4caf50', mr: 1 }} />
                <Typography variant="subtitle1" sx={{ fontWeight: 'bold' }}>
                  Recommendations
                </Typography>
              </Box>
              {recommendations.slice(0, 2).map((recommendation, index) => (
                <Typography key={index} variant="body2" sx={{ mb: 1 }}>
                  • {recommendation}
                </Typography>
              ))}
            </Box>
          )}
        </CardContent>
      </Card>
    );
  };

  // Early returns for loading and error states
  if (loading) {
    return (
      <Box sx={{ p: 3, backgroundColor: '#f5f5f5', minHeight: '100vh' }}>
        <LinearProgress />
        <Typography variant="h6" sx={{ mt: 2, textAlign: 'center' }}>
          Loading environmental dashboard data...
        </Typography>
      </Box>
    );
  }

  if (error) {
    return (
      <Box sx={{ p: 3, backgroundColor: '#f5f5f5', minHeight: '100vh' }}>
        <Alert severity="error" sx={{ mb: 2 }}>
          Failed to load environmental dashboard data: {error}
        </Alert>
        <Button 
          variant="contained" 
          onClick={fetchEnvironmentalData}
          sx={{ mt: 2 }}
        >
          Retry
        </Button>
      </Box>
    );
  }

  // Calculate overall metrics for header
  const overallCO2Reduction = calculateCO2Emissions(electricityData) + calculateCO2Emissions(waterData);
  const overallRiskLevel = (electricityData?.risks?.length || 0) + (waterData?.risks?.length || 0) + (wasteData?.risks?.length || 0);
  const averageScore = Math.round(((electricityData?.goals?.length || 0) + (waterData?.goals?.length || 0) + (wasteData?.goals?.length || 0)) * 33.33);

  return (
    <Box sx={{ p: 3, backgroundColor: '#f5f5f5', minHeight: '100vh' }}>
      {/* Global Header */}
      <Box sx={{ mb: 4, backgroundColor: 'white', p: 3, borderRadius: 2, boxShadow: 1 }}>
        <Grid container spacing={3} alignItems="center">
          <Grid item xs={12} md={6}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
              <Typography variant="h5" sx={{ fontWeight: 'bold' }}>
                Environmental Dashboard
              </Typography>
              <Chip 
                label={locationHierarchy[selectedLocation]}
                variant="outlined"
                size="small"
              />
            </Box>
          </Grid>
          <Grid item xs={12} md={6}>
            <Box sx={{ display: 'flex', gap: 2, justifyContent: 'flex-end' }}>
              <FormControl size="small" sx={{ minWidth: 200 }}>
                <InputLabel>Location</InputLabel>
                <Select
                  value={selectedLocation}
                  label="Location"
                  onChange={(e) => setSelectedLocation(e.target.value)}
                >
                  <MenuItem value="algonquin_il">Algonquin Site (IL)</MenuItem>
                  <MenuItem value="houston_tx">Houston Site (TX)</MenuItem>
                </Select>
              </FormControl>
              <ButtonGroup size="small">
                <Button 
                  variant={selectedDateRange === '30days' ? 'contained' : 'outlined'}
                  onClick={() => setSelectedDateRange('30days')}
                >
                  Last 30 Days
                </Button>
                <Button 
                  variant={selectedDateRange === 'quarter' ? 'contained' : 'outlined'}
                  onClick={() => setSelectedDateRange('quarter')}
                >
                  This Quarter
                </Button>
                <Button 
                  variant={selectedDateRange === 'custom' ? 'contained' : 'outlined'}
                  onClick={() => setSelectedDateRange('custom')}
                >
                  Custom Range
                </Button>
              </ButtonGroup>
            </Box>
          </Grid>
        </Grid>
      </Box>

      {/* Overall Environmental Scorecard */}
      <Card sx={{ mb: 4, boxShadow: 3 }}>
        <CardContent sx={{ p: 4 }}>
          <Typography variant="h5" sx={{ mb: 3, fontWeight: 'bold' }}>
            Overall Environmental Scorecard
          </Typography>
          <Grid container spacing={4}>
            <Grid item xs={12} md={3}>
              <Box sx={{ textAlign: 'center' }}>
                <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', mb: 1 }}>
                  <CheckCircle sx={{ color: '#4caf50', mr: 1 }} />
                  <Typography variant="h3" sx={{ fontWeight: 'bold', color: '#4caf50' }}>
                    {averageScore}%
                  </Typography>
                </Box>
                <Typography variant="body1" color="textSecondary">
                  Environmental Score
                </Typography>
              </Box>
            </Grid>
            <Grid item xs={12} md={3}>
              <Box sx={{ textAlign: 'center' }}>
                <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', mb: 1 }}>
                  <MonetizationOn sx={{ color: '#4caf50', mr: 1 }} />
                  <Typography variant="h3" sx={{ fontWeight: 'bold', color: '#4caf50' }}>
                    ${Math.round(overallCO2Reduction * 0.5)}K
                  </Typography>
                </Box>
                <Typography variant="body1" color="textSecondary">
                  Estimated Cost Savings
                </Typography>
              </Box>
            </Grid>
            <Grid item xs={12} md={3}>
              <Box sx={{ textAlign: 'center' }}>
                <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', mb: 1 }}>
                  <Park sx={{ color: '#4caf50', mr: 1 }} />
                  <Typography variant="h3" sx={{ fontWeight: 'bold', color: '#4caf50' }}>
                    {overallCO2Reduction}
                  </Typography>
                </Box>
                <Typography variant="body1" color="textSecondary">
                  Tons CO₂ Reduced
                </Typography>
              </Box>
            </Grid>
            <Grid item xs={12} md={3}>
              <Box sx={{ textAlign: 'center' }}>
                <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', mb: 1 }}>
                  <Warning sx={{ color: overallRiskLevel > 5 ? '#f44336' : '#ff9800', mr: 1 }} />
                  <Chip 
                    label={overallRiskLevel > 5 ? "High Risk" : overallRiskLevel > 2 ? "Medium Risk" : "Low Risk"}
                    color={overallRiskLevel > 5 ? 'error' : overallRiskLevel > 2 ? 'warning' : 'success'}
                    variant="filled"
                    sx={{ fontSize: '1rem', px: 2, py: 1 }}
                  />
                </Box>
                <Typography variant="body1" color="textSecondary">
                  Overall Risk Level
                </Typography>
              </Box>
            </Grid>
          </Grid>
        </CardContent>
      </Card>

      {/* Three Environmental Cards */}
      <Grid container spacing={3}>
        {/* CO2 Emissions Card (Electricity) */}
        <Grid item xs={12} md={4}>
          {renderEnvironmentalCard(
            electricityData,
            'CO2 Emissions',
            <Bolt sx={{ color: '#ff9800', mr: 2, fontSize: 30 }} />,
            '#ff9800',
            'electricity'
          )}
        </Grid>

        {/* Water Consumption Card */}
        <Grid item xs={12} md={4}>
          {renderEnvironmentalCard(
            waterData,
            'Water Consumption',
            <Water sx={{ color: '#2196f3', mr: 2, fontSize: 30 }} />,
            '#2196f3',
            'water'
          )}
        </Grid>

        {/* Waste Generation Card */}
        <Grid item xs={12} md={4}>
          {renderEnvironmentalCard(
            wasteData,
            'Waste Generation',
            <Delete sx={{ color: '#4caf50', mr: 2, fontSize: 30 }} />,
            '#4caf50',
            'waste'
          )}
        </Grid>
      </Grid>
    </Box>
  );
};

export default AnalyticsExecutive;