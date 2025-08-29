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
  const [selectedLocation, setSelectedLocation] = useState('algonquin');
  const [selectedDateRange, setSelectedDateRange] = useState('30days');
  const [dashboardData, setDashboardData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Location hierarchy mapping for display
  const locationHierarchy = {
    'global': 'Global View',
    'northamerica': 'Global View → North America',
    'illinois': 'Global View → North America → Illinois',
    'algonquin': 'Global View → North America → Illinois → Algonquin Site',
    'texas': 'Global View → North America → Texas',
    'houston': 'Global View → North America → Texas → Houston'
  };

  // Fetch dashboard data from API
  const fetchDashboardData = async () => {
    setLoading(true);
    setError(null);
    setDashboardData(null); // Clear data during fetch to prevent stale data access
    
    try {
      const response = await fetch(
        `${API_ENDPOINTS.executiveDashboard}?location=${selectedLocation}&dateRange=${selectedDateRange}`
      );
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      
      // Transform v2 API data to component format
      const transformedData = {
        globalScorecard: {
          ehsScore: data.summary?.overall_health_score || 0,
          costSavings: 0 // No cost savings in v2 API yet
        },
        riskAssessment: data.risk_assessment || {},
        kpis: data.kpis || {},
        recommendations: data.recommendations || {},
        // Add mock data for missing sections to prevent errors
        electricity: {
          goal: { 
            actual: data.kpis?.metrics?.overall_risk_score?.value || 0,
            target: 100, 
            unit: 'Score' 
          },
          facts: {
            consumption: `Risk Score: ${data.kpis?.metrics?.overall_risk_score?.value || 0}`,
            cost: `Status: ${data.kpis?.metrics?.overall_risk_score?.status || 'N/A'}`,
            co2: 'Risk assessment data available'
          },
          analysis: `Risk Level: ${data.risk_assessment?.overall_risk_level || 'N/A'}. Total assessments: ${data.risk_assessment?.total_assessments || 0}`,
          recommendation: data.recommendations?.risk_based_recommendations?.[0]?.description || 'No recommendations available'
        },
        water: {
          goal: { 
            actual: data.kpis?.metrics?.incident_rate?.value || 0,
            target: 5, 
            unit: 'Rate' 
          },
          facts: {
            consumption: `Incident Rate: ${data.kpis?.metrics?.incident_rate?.value || 0}`,
            cost: `Total Incidents: ${data.summary?.incidents?.total || 0}`,
            co2: 'Safety metrics monitoring'
          },
          analysis: `Incident rate tracking shows ${data.kpis?.metrics?.incident_rate?.status || 'unknown'} status`,
          recommendation: 'Continue safety monitoring and training programs'
        },
        waste: {
          goal: { 
            actual: data.kpis?.metrics?.audit_pass_rate?.value || 0,
            target: 100, 
            unit: '%' 
          },
          facts: {
            generation: `Audit Pass Rate: ${data.kpis?.metrics?.audit_pass_rate?.value || 0}%`,
            cost: `Training Completion: ${data.summary?.compliance?.training_completion || 0}%`,
            co2: 'Compliance monitoring active'
          },
          analysis: `Audit performance shows ${data.kpis?.metrics?.audit_pass_rate?.status || 'unknown'} status`,
          recommendation: 'Maintain compliance standards and improve audit processes'
        },
        compliance: {
          goal: { 
            actual: data.summary?.compliance?.audit_pass_rate || 0,
            target: 100, 
            unit: '%' 
          },
          facts: [`Audit Pass Rate: ${data.summary?.compliance?.audit_pass_rate || 0}%`],
          analysis: 'Compliance metrics available',
          recommendation: 'Maintain compliance standards'
        }
      };

      setDashboardData(transformedData);
    } catch (err) {
      console.error('Error fetching dashboard data:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDashboardData();
  }, [selectedLocation, selectedDateRange]);

  const getScoreColor = (score, target) => {
    if (score >= target) return '#4caf50';
    if (score >= target * 0.8) return '#ff9800';
    return '#f44336';
  };

  const getTrendIcon = (trend) => {
    // Handle string trends
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

    // Handle array trends (original logic)
    if (!Array.isArray(trend) || trend.length < 2) return <Info />;
    
    const recent = trend.slice(-3).reduce((a, b) => a + b, 0) / 3;
    const previous = trend.slice(-6, -3).reduce((a, b) => a + b, 0) / 3;
    
    if (recent > previous) return <TrendingUp sx={{ color: '#4caf50' }} />;
    return <TrendingDown sx={{ color: '#f44336' }} />;
  };

  const renderSparkline = (data) => {
    // Only render sparkline for array data
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

  // Early returns for loading and error states
  if (loading) {
    return (
      <Box sx={{ p: 3, backgroundColor: '#f5f5f5', minHeight: '100vh' }}>
        <LinearProgress />
        <Typography variant="h6" sx={{ mt: 2, textAlign: 'center' }}>
          Loading dashboard data...
        </Typography>
      </Box>
    );
  }

  if (error) {
    return (
      <Box sx={{ p: 3, backgroundColor: '#f5f5f5', minHeight: '100vh' }}>
        <Alert severity="error" sx={{ mb: 2 }}>
          Failed to load dashboard data: {error}
        </Alert>
        <Button 
          variant="contained" 
          onClick={fetchDashboardData}
          sx={{ mt: 2 }}
        >
          Retry
        </Button>
      </Box>
    );
  }

  // Early return if dashboardData is null or undefined
  if (!dashboardData) {
    return (
      <Box sx={{ p: 3, backgroundColor: '#f5f5f5', minHeight: '100vh' }}>
        <LinearProgress />
        <Typography variant="h6" sx={{ mt: 2, textAlign: 'center' }}>
          Loading dashboard data...
        </Typography>
      </Box>
    );
  }

  // Additional safety checks for nested data structures
  const hasGlobalScorecard = dashboardData.globalScorecard && 
    typeof dashboardData.globalScorecard.ehsScore !== 'undefined';
  
  const hasElectricityData = dashboardData.electricity &&
    dashboardData.electricity.goal &&
    dashboardData.electricity.facts &&
    dashboardData.electricity.analysis &&
    dashboardData.electricity.recommendation;
    
  const hasWaterData = dashboardData.water &&
    dashboardData.water.goal &&
    dashboardData.water.facts &&
    dashboardData.water.analysis &&
    dashboardData.water.recommendation;
    
  const hasWasteData = dashboardData.waste &&
    dashboardData.waste.goal &&
    dashboardData.waste.facts &&
    dashboardData.waste.analysis &&
    dashboardData.waste.recommendation;

  return (
    <Box sx={{ p: 3, backgroundColor: '#f5f5f5', minHeight: '100vh' }}>
      {/* Global Header */}
      <Box sx={{ mb: 4, backgroundColor: 'white', p: 3, borderRadius: 2, boxShadow: 1 }}>
        <Grid container spacing={3} alignItems="center">
          <Grid item xs={12} md={6}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
              <Typography variant="h5" sx={{ fontWeight: 'bold' }}>
                EHS Executive Dashboard
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
                  <MenuItem value="global">Global View</MenuItem>
                  <ListSubheader>North America</ListSubheader>
                  <ListSubheader sx={{ pl: 4 }}>Illinois</ListSubheader>
                  <MenuItem value="algonquin" sx={{ pl: 6 }}>Algonquin Site</MenuItem>
                  <ListSubheader sx={{ pl: 4 }}>Texas</ListSubheader>
                  <MenuItem value="houston" sx={{ pl: 6 }}>Houston</MenuItem>
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

      {/* Overall EHS Scorecard */}
      {hasGlobalScorecard && (
        <Card sx={{ mb: 4, boxShadow: 3 }}>
          <CardContent sx={{ p: 4 }}>
            <Typography variant="h5" sx={{ mb: 3, fontWeight: 'bold' }}>
              Overall EHS Scorecard
            </Typography>
            <Grid container spacing={4}>
              <Grid item xs={12} md={3}>
                <Box sx={{ textAlign: 'center' }}>
                  <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', mb: 1 }}>
                    <CheckCircle sx={{ color: '#4caf50', mr: 1 }} />
                    <Typography variant="h3" sx={{ fontWeight: 'bold', color: '#4caf50' }}>
                      {dashboardData.globalScorecard.ehsScore}%
                    </Typography>
                  </Box>
                  <Typography variant="body1" color="textSecondary">
                    Total EHS Score
                  </Typography>
                </Box>
              </Grid>
              <Grid item xs={12} md={3}>
                <Box sx={{ textAlign: 'center' }}>
                  <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', mb: 1 }}>
                    <MonetizationOn sx={{ color: '#4caf50', mr: 1 }} />
                    <Typography variant="h3" sx={{ fontWeight: 'bold', color: '#4caf50' }}>
                      ${dashboardData.globalScorecard.costSavings ? 
                        (dashboardData.globalScorecard.costSavings / 1000).toFixed(0) : '0'}K
                    </Typography>
                  </Box>
                  <Typography variant="body1" color="textSecondary">
                    Total Cost Savings
                  </Typography>
                </Box>
              </Grid>
              <Grid item xs={12} md={3}>
                <Box sx={{ textAlign: 'center' }}>
                  <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', mb: 1 }}>
                    <Park sx={{ color: '#4caf50', mr: 1 }} />
                    <Typography variant="h3" sx={{ fontWeight: 'bold', color: '#4caf50' }}>
                      {dashboardData.globalScorecard.carbonFootprint || '0'}
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
                    <Warning sx={{ color: '#ff9800', mr: 1 }} />
                    <Chip 
                      label={dashboardData.riskAssessment?.overall_risk_level || "Medium Risk"}
                      color={dashboardData.riskAssessment?.overall_risk_level === 'LOW' ? 'success' : 'warning'}
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
      )}

      {/* Three Thematic Detail Cards */}
      {(hasElectricityData || hasWaterData || hasWasteData) && (
        <Grid container spacing={3}>
          {/* Risk Assessment Card (formerly Electricity) */}
          {hasElectricityData && (
            <Grid item xs={12} md={4}>
              <Card sx={{ height: '100%', boxShadow: 3 }}>
                <CardContent sx={{ p: 3 }}>
                  <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
                    <Warning sx={{ color: '#ff9800', mr: 2, fontSize: 30 }} />
                    <Typography variant="h6" sx={{ fontWeight: 'bold' }}>
                      Risk Assessment
                    </Typography>
                  </Box>
                  
                  {/* Goal Section */}
                  <Box sx={{ mb: 3, p: 2, backgroundColor: '#f8f9fa', borderRadius: 1 }}>
                    <Typography variant="subtitle1" sx={{ mb: 2, fontWeight: 'bold' }}>
                      Risk Score
                    </Typography>
                    {renderGauge(
                      dashboardData.electricity.goal.actual,
                      dashboardData.electricity.goal.target,
                      dashboardData.electricity.goal.unit
                    )}
                    <Typography variant="body2" sx={{ textAlign: 'center', mt: 1 }}>
                      {dashboardData.electricity.goal.unit}
                    </Typography>
                  </Box>

                  {/* Facts Section */}
                  <Box sx={{ mb: 3, p: 2, backgroundColor: '#e3f2fd', borderRadius: 1 }}>
                    <Typography variant="subtitle1" sx={{ mb: 2, fontWeight: 'bold' }}>
                      Key Facts
                    </Typography>
                    <Typography variant="body2" sx={{ mb: 1 }}>
                      <strong>Risk Score:</strong> {dashboardData.electricity.facts.consumption}
                    </Typography>
                    <Typography variant="body2" sx={{ mb: 1 }}>
                      <strong>Status:</strong> {dashboardData.electricity.facts.cost}
                    </Typography>
                    <Typography variant="body2">
                      <strong>Assessment:</strong> {dashboardData.electricity.facts.co2}
                    </Typography>
                  </Box>

                  {/* Analysis Section */}
                  <Box sx={{ mb: 3, p: 2, backgroundColor: '#fff3e0', borderRadius: 1 }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                      <Typography variant="subtitle1" sx={{ fontWeight: 'bold', flexGrow: 1 }}>
                        Risk Analysis
                      </Typography>
                      {dashboardData.electricity.trend && getTrendIcon(dashboardData.electricity.trend)}
                    </Box>
                    {Array.isArray(dashboardData.electricity.trend) && (
                      <Box sx={{ mb: 2 }}>
                        {renderSparkline(dashboardData.electricity.trend)}
                      </Box>
                    )}
                    <Typography variant="body2">
                      {dashboardData.electricity.analysis}
                    </Typography>
                  </Box>

                  {/* Recommendation Section */}
                  <Box sx={{ p: 2, backgroundColor: '#e8f5e8', borderRadius: 1 }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                      <LightbulbOutlined sx={{ color: '#4caf50', mr: 1 }} />
                      <Typography variant="subtitle1" sx={{ fontWeight: 'bold' }}>
                        Recommendation
                      </Typography>
                    </Box>
                    <Typography variant="body2">
                      {dashboardData.electricity.recommendation}
                    </Typography>
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          )}

          {/* Safety Incidents Card (formerly Water) */}
          {hasWaterData && (
            <Grid item xs={12} md={4}>
              <Card sx={{ height: '100%', boxShadow: 3 }}>
                <CardContent sx={{ p: 3 }}>
                  <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
                    <CheckCircle sx={{ color: '#2196f3', mr: 2, fontSize: 30 }} />
                    <Typography variant="h6" sx={{ fontWeight: 'bold' }}>
                      Safety Incidents
                    </Typography>
                  </Box>
                  
                  {/* Goal Section */}
                  <Box sx={{ mb: 3, p: 2, backgroundColor: '#f8f9fa', borderRadius: 1 }}>
                    <Typography variant="subtitle1" sx={{ mb: 2, fontWeight: 'bold' }}>
                      Incident Rate
                    </Typography>
                    {renderGauge(
                      dashboardData.water.goal.actual,
                      dashboardData.water.goal.target,
                      dashboardData.water.goal.unit
                    )}
                    <Typography variant="body2" sx={{ textAlign: 'center', mt: 1 }}>
                      {dashboardData.water.goal.unit}
                    </Typography>
                  </Box>

                  {/* Facts Section */}
                  <Box sx={{ mb: 3, p: 2, backgroundColor: '#e3f2fd', borderRadius: 1 }}>
                    <Typography variant="subtitle1" sx={{ mb: 2, fontWeight: 'bold' }}>
                      Key Facts
                    </Typography>
                    <Typography variant="body2" sx={{ mb: 1 }}>
                      <strong>Rate:</strong> {dashboardData.water.facts.consumption}
                    </Typography>
                    <Typography variant="body2" sx={{ mb: 1 }}>
                      <strong>Total:</strong> {dashboardData.water.facts.cost}
                    </Typography>
                    <Typography variant="body2">
                      <strong>Status:</strong> {dashboardData.water.facts.co2}
                    </Typography>
                  </Box>

                  {/* Analysis Section */}
                  <Box sx={{ mb: 3, p: 2, backgroundColor: '#fff3e0', borderRadius: 1 }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                      <Typography variant="subtitle1" sx={{ fontWeight: 'bold', flexGrow: 1 }}>
                        Safety Analysis
                      </Typography>
                      {dashboardData.water.trend && getTrendIcon(dashboardData.water.trend)}
                    </Box>
                    {Array.isArray(dashboardData.water.trend) && (
                      <Box sx={{ mb: 2 }}>
                        {renderSparkline(dashboardData.water.trend)}
                      </Box>
                    )}
                    <Typography variant="body2">
                      {dashboardData.water.analysis}
                    </Typography>
                  </Box>

                  {/* Recommendation Section */}
                  <Box sx={{ p: 2, backgroundColor: '#e8f5e8', borderRadius: 1 }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                      <LightbulbOutlined sx={{ color: '#4caf50', mr: 1 }} />
                      <Typography variant="subtitle1" sx={{ fontWeight: 'bold' }}>
                        Recommendation
                      </Typography>
                    </Box>
                    <Typography variant="body2">
                      {dashboardData.water.recommendation}
                    </Typography>
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          )}

          {/* Compliance Card (formerly Waste) */}
          {hasWasteData && (
            <Grid item xs={12} md={4}>
              <Card sx={{ height: '100%', boxShadow: 3 }}>
                <CardContent sx={{ p: 3 }}>
                  <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
                    <Info sx={{ color: '#4caf50', mr: 2, fontSize: 30 }} />
                    <Typography variant="h6" sx={{ fontWeight: 'bold' }}>
                      Compliance
                    </Typography>
                  </Box>
                  
                  {/* Goal Section */}
                  <Box sx={{ mb: 3, p: 2, backgroundColor: '#f8f9fa', borderRadius: 1 }}>
                    <Typography variant="subtitle1" sx={{ mb: 2, fontWeight: 'bold' }}>
                      Audit Pass Rate
                    </Typography>
                    {renderGauge(
                      dashboardData.waste.goal.actual,
                      dashboardData.waste.goal.target,
                      dashboardData.waste.goal.unit
                    )}
                    <Typography variant="body2" sx={{ textAlign: 'center', mt: 1 }}>
                      {dashboardData.waste.goal.unit}
                    </Typography>
                  </Box>

                  {/* Facts Section */}
                  <Box sx={{ mb: 3, p: 2, backgroundColor: '#e3f2fd', borderRadius: 1 }}>
                    <Typography variant="subtitle1" sx={{ mb: 2, fontWeight: 'bold' }}>
                      Key Facts
                    </Typography>
                    <Typography variant="body2" sx={{ mb: 1 }}>
                      <strong>Pass Rate:</strong> {dashboardData.waste.facts.generation}
                    </Typography>
                    <Typography variant="body2" sx={{ mb: 1 }}>
                      <strong>Training:</strong> {dashboardData.waste.facts.cost}
                    </Typography>
                    <Typography variant="body2">
                      <strong>Status:</strong> {dashboardData.waste.facts.co2}
                    </Typography>
                  </Box>

                  {/* Analysis Section */}
                  <Box sx={{ mb: 3, p: 2, backgroundColor: '#fff3e0', borderRadius: 1 }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                      <Typography variant="subtitle1" sx={{ fontWeight: 'bold', flexGrow: 1 }}>
                        Compliance Analysis
                      </Typography>
                      {dashboardData.waste.trend && getTrendIcon(dashboardData.waste.trend)}
                    </Box>
                    {Array.isArray(dashboardData.waste.trend) && (
                      <Box sx={{ mb: 2 }}>
                        {renderSparkline(dashboardData.waste.trend)}
                      </Box>
                    )}
                    <Typography variant="body2">
                      {dashboardData.waste.analysis}
                    </Typography>
                  </Box>

                  {/* Recommendation Section */}
                  <Box sx={{ p: 2, backgroundColor: '#e8f5e8', borderRadius: 1 }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                      <LightbulbOutlined sx={{ color: '#4caf50', mr: 1 }} />
                      <Typography variant="subtitle1" sx={{ fontWeight: 'bold' }}>
                        Recommendation
                      </Typography>
                    </Box>
                    <Typography variant="body2">
                      {dashboardData.waste.recommendation}
                    </Typography>
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          )}
        </Grid>
      )}
    </Box>
  );
};

export default AnalyticsExecutive;