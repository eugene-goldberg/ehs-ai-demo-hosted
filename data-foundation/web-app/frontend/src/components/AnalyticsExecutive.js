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
  ButtonGroup
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

const AnalyticsExecutive = () => {
  const [selectedLocation, setSelectedLocation] = useState('algonquin');
  const [selectedDateRange, setSelectedDateRange] = useState('30days');
  const [dashboardData, setDashboardData] = useState(null);
  const [loading, setLoading] = useState(false);

  // Mock executive dashboard data
  const executiveDashboardData = {
    globalScorecard: {
      ehsScore: 87,
      costSavings: 245000,
      carbonFootprint: 1250,
      riskLevel: 'medium'
    },
    electricity: {
      goal: { actual: 85, target: 80, unit: '% efficiency' },
      facts: {
        consumption: '2.4M kWh',
        cost: '$340K',
        co2: '720 tons CO₂'
      },
      trend: [78, 82, 79, 85, 87, 85, 88, 85],
      analysis: 'Energy consumption trending up 8% this quarter due to increased production',
      recommendation: 'Install LED lighting and smart HVAC controls to reduce consumption by 15%'
    },
    water: {
      goal: { actual: 72, target: 85, unit: '% efficiency' },
      facts: {
        consumption: '150K gallons',
        cost: '$45K',
        co2: '85 tons CO₂'
      },
      trend: [85, 82, 78, 72, 69, 72, 75, 72],
      analysis: 'Water efficiency below target due to cooling system inefficiencies',
      recommendation: 'Upgrade cooling tower systems and implement water recycling program'
    },
    waste: {
      goal: { actual: 92, target: 90, unit: '% diversion' },
      facts: {
        generation: '45 tons',
        cost: '$12K saved',
        co2: '120 tons CO₂ avoided'
      },
      trend: [88, 90, 89, 92, 94, 92, 91, 92],
      analysis: 'Waste diversion exceeding targets with strong recycling program performance',
      recommendation: 'Expand composting program to achieve 95% diversion rate'
    }
  };

  useEffect(() => {
    setDashboardData(executiveDashboardData);
  }, [selectedLocation, selectedDateRange]);

  const getScoreColor = (score, target) => {
    if (score >= target) return '#4caf50';
    if (score >= target * 0.8) return '#ff9800';
    return '#f44336';
  };

  const getTrendIcon = (trend) => {
    if (!trend || trend.length < 2) return <Info />;
    const recent = trend.slice(-3).reduce((a, b) => a + b, 0) / 3;
    const previous = trend.slice(-6, -3).reduce((a, b) => a + b, 0) / 3;
    
    if (recent > previous) return <TrendingUp sx={{ color: '#4caf50' }} />;
    return <TrendingDown sx={{ color: '#f44336' }} />;
  };

  const renderSparkline = (data) => {
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

  if (loading) {
    return <LinearProgress />;
  }

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
                label="Global View → North America → Illinois → Algonquin Site"
                variant="outlined"
                size="small"
              />
            </Box>
          </Grid>
          <Grid item xs={12} md={6}>
            <Box sx={{ display: 'flex', gap: 2, justifyContent: 'flex-end' }}>
              <FormControl size="small" sx={{ minWidth: 150 }}>
                <InputLabel>Location</InputLabel>
                <Select
                  value={selectedLocation}
                  label="Location"
                  onChange={(e) => setSelectedLocation(e.target.value)}
                >
                  <MenuItem value="global">Global View</MenuItem>
                  <MenuItem value="northamerica">North America</MenuItem>
                  <MenuItem value="illinois">Illinois</MenuItem>
                  <MenuItem value="algonquin">Algonquin Site</MenuItem>
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
      {dashboardData && (
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
                      ${(dashboardData.globalScorecard.costSavings / 1000).toFixed(0)}K
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
                      {dashboardData.globalScorecard.carbonFootprint}
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
                      label="Medium Risk"
                      color="warning"
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
      {dashboardData && (
        <Grid container spacing={3}>
          {/* Electricity Consumption Card */}
          <Grid item xs={12} md={4}>
            <Card sx={{ height: '100%', boxShadow: 3 }}>
              <CardContent sx={{ p: 3 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
                  <Bolt sx={{ color: '#ff9800', mr: 2, fontSize: 30 }} />
                  <Typography variant="h6" sx={{ fontWeight: 'bold' }}>
                    Electricity Consumption
                  </Typography>
                </Box>
                
                {/* Goal Section */}
                <Box sx={{ mb: 3, p: 2, backgroundColor: '#f8f9fa', borderRadius: 1 }}>
                  <Typography variant="subtitle1" sx={{ mb: 2, fontWeight: 'bold' }}>
                    Goal Performance
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
                    <strong>Consumption:</strong> {dashboardData.electricity.facts.consumption}
                  </Typography>
                  <Typography variant="body2" sx={{ mb: 1 }}>
                    <strong>Cost:</strong> {dashboardData.electricity.facts.cost}
                  </Typography>
                  <Typography variant="body2">
                    <strong>CO₂ Impact:</strong> {dashboardData.electricity.facts.co2}
                  </Typography>
                </Box>

                {/* Analysis Section */}
                <Box sx={{ mb: 3, p: 2, backgroundColor: '#fff3e0', borderRadius: 1 }}>
                  <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                    <Typography variant="subtitle1" sx={{ fontWeight: 'bold', flexGrow: 1 }}>
                      Trend Analysis
                    </Typography>
                    {getTrendIcon(dashboardData.electricity.trend)}
                  </Box>
                  <Box sx={{ mb: 2 }}>
                    {renderSparkline(dashboardData.electricity.trend)}
                  </Box>
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

          {/* Water Consumption Card */}
          <Grid item xs={12} md={4}>
            <Card sx={{ height: '100%', boxShadow: 3 }}>
              <CardContent sx={{ p: 3 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
                  <Water sx={{ color: '#2196f3', mr: 2, fontSize: 30 }} />
                  <Typography variant="h6" sx={{ fontWeight: 'bold' }}>
                    Water Consumption
                  </Typography>
                </Box>
                
                {/* Goal Section */}
                <Box sx={{ mb: 3, p: 2, backgroundColor: '#f8f9fa', borderRadius: 1 }}>
                  <Typography variant="subtitle1" sx={{ mb: 2, fontWeight: 'bold' }}>
                    Goal Performance
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
                    <strong>Consumption:</strong> {dashboardData.water.facts.consumption}
                  </Typography>
                  <Typography variant="body2" sx={{ mb: 1 }}>
                    <strong>Cost:</strong> {dashboardData.water.facts.cost}
                  </Typography>
                  <Typography variant="body2">
                    <strong>CO₂ Impact:</strong> {dashboardData.water.facts.co2}
                  </Typography>
                </Box>

                {/* Analysis Section */}
                <Box sx={{ mb: 3, p: 2, backgroundColor: '#fff3e0', borderRadius: 1 }}>
                  <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                    <Typography variant="subtitle1" sx={{ fontWeight: 'bold', flexGrow: 1 }}>
                      Trend Analysis
                    </Typography>
                    {getTrendIcon(dashboardData.water.trend)}
                  </Box>
                  <Box sx={{ mb: 2 }}>
                    {renderSparkline(dashboardData.water.trend)}
                  </Box>
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

          {/* Waste Generation Card */}
          <Grid item xs={12} md={4}>
            <Card sx={{ height: '100%', boxShadow: 3 }}>
              <CardContent sx={{ p: 3 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
                  <Delete sx={{ color: '#4caf50', mr: 2, fontSize: 30 }} />
                  <Typography variant="h6" sx={{ fontWeight: 'bold' }}>
                    Waste Generation
                  </Typography>
                </Box>
                
                {/* Goal Section */}
                <Box sx={{ mb: 3, p: 2, backgroundColor: '#f8f9fa', borderRadius: 1 }}>
                  <Typography variant="subtitle1" sx={{ mb: 2, fontWeight: 'bold' }}>
                    Goal Performance
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
                    <strong>Generation:</strong> {dashboardData.waste.facts.generation}
                  </Typography>
                  <Typography variant="body2" sx={{ mb: 1 }}>
                    <strong>Cost Impact:</strong> {dashboardData.waste.facts.cost}
                  </Typography>
                  <Typography variant="body2">
                    <strong>CO₂ Avoided:</strong> {dashboardData.waste.facts.co2}
                  </Typography>
                </Box>

                {/* Analysis Section */}
                <Box sx={{ mb: 3, p: 2, backgroundColor: '#fff3e0', borderRadius: 1 }}>
                  <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                    <Typography variant="subtitle1" sx={{ fontWeight: 'bold', flexGrow: 1 }}>
                      Trend Analysis
                    </Typography>
                    {getTrendIcon(dashboardData.waste.trend)}
                  </Box>
                  <Box sx={{ mb: 2 }}>
                    {renderSparkline(dashboardData.waste.trend)}
                  </Box>
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
        </Grid>
      )}
    </Box>
  );
};

export default AnalyticsExecutive;