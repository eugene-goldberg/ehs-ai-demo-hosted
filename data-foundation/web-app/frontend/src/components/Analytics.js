import React, { useState, useEffect } from 'react';
import axios from 'axios';

const Analytics = () => {
  const [dashboardData, setDashboardData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedMetric, setSelectedMetric] = useState('incident_trends');
  const [analyticsData, setAnalyticsData] = useState(null);
  const [dateRange, setDateRange] = useState('Aug 25 - Sep 25');

  // Mock data for risk-focused EHS dashboard
  const mockRiskData = {
    riskAlerts: {
      critical: { value: 3, label: 'Critical', color: '#dc3545' },
      high: { value: 7, label: 'High', color: '#fd7e14' },
      medium: { value: 12, label: 'Medium', color: '#ffc107' },
      actions: { value: 22, label: 'Actions', color: '#007bff' }
    },
    waterConsumption: {
      trend: [
        { month: 'May', consumption: 85, target: 80 },
        { month: 'Jun', consumption: 90, target: 80 },
        { month: 'Jul', consumption: 95, target: 80 },
        { month: 'Aug', consumption: 100, target: 80 }
      ],
      riskText: "RISK: Water consumption at Facility Mumbai has increased 25% over the last quarter, threatening annual reduction goals of 15%",
      recommendationText: "RECOMMENDATION: Review water pump efficiency at Facility Mumbai. Replace pumps #3 and #5 which are operating at 60% efficiency. Estimated savings: 2,500 gallons/month"
    },
    electricityConsumption: {
      trend: [
        { month: 'May', consumption: 320, peakDemand: 380 },
        { month: 'Jun', consumption: 340, peakDemand: 400 },
        { month: 'Jul', consumption: 365, peakDemand: 420 },
        { month: 'Aug', consumption: 385, peakDemand: 460 }
      ],
      riskText: "RISK: Peak demand charges at Facility Shanghai exceeded budget by 40% in August due to inefficient HVAC scheduling",
      recommendationText: "RECOMMENDATION: Implement smart HVAC controls with occupancy sensors. Adjust production schedules to avoid peak hours (2-6 PM). Potential savings: $15,000/month"
    },
    wasteGeneration: {
      trend: [
        { month: 'May', waste: 45, hazardous: 8 },
        { month: 'Jun', waste: 48, hazardous: 9 },
        { month: 'Jul', waste: 52, hazardous: 12 },
        { month: 'Aug', waste: 55, hazardous: 15 }
      ],
      riskText: "RISK: Hazardous waste disposal costs increased 35% due to improper segregation at Facility Berlin",
      recommendationText: "RECOMMENDATION: Conduct waste segregation training for all staff. Install color-coded bins at 15 key locations. Expected reduction: 20% in disposal costs"
    },
    riskMatrix: [
      { name: 'Water Pump Failure', impact: 85, likelihood: 75, type: 'Equipment' },
      { name: 'HVAC Overuse', impact: 60, likelihood: 90, type: 'Energy' },
      { name: 'Waste Segregation', impact: 70, likelihood: 80, type: 'Waste' },
      { name: 'Compliance Audit', impact: 95, likelihood: 40, type: 'Regulatory' },
      { name: 'Emergency Response', impact: 90, likelihood: 20, type: 'Safety' }
    ],
    actionItems: [
      { id: 1, task: 'Replace water pumps #3 and #5', facility: 'Mumbai', deadline: '2024-09-15', priority: 'High', status: 30 },
      { id: 2, task: 'Install smart HVAC controls', facility: 'Shanghai', deadline: '2024-09-30', priority: 'Critical', status: 10 },
      { id: 3, task: 'Waste segregation training', facility: 'Berlin', deadline: '2024-09-20', priority: 'Medium', status: 60 },
      { id: 4, task: 'Compliance documentation review', facility: 'London', deadline: '2024-10-05', priority: 'High', status: 80 },
      { id: 5, task: 'Emergency response drill', facility: 'Tokyo', deadline: '2024-09-25', priority: 'Medium', status: 0 }
    ]
  };

  useEffect(() => {
    fetchDashboardData();
  }, []);

  const fetchDashboardData = async () => {
    try {
      setLoading(true);
      const response = await axios.get('http://localhost:8001/api/analytics/dashboard');
      setDashboardData(response.data);
      setError(null);
    } catch (err) {
      setError('Failed to fetch analytics data. Please try again.');
      console.error('Error fetching dashboard data:', err);
    } finally {
      setLoading(false);
    }
  };

  const fetchAnalyticsQuery = async (metricType) => {
    try {
      const response = await axios.post('http://localhost:8001/api/analytics/query', {
        metric_type: metricType,
        date_range: {
          start: '2024-01-01',
          end: '2024-12-31'
        }
      });
      setAnalyticsData(response.data);
    } catch (err) {
      console.error('Error fetching analytics query:', err);
    }
  };

  useEffect(() => {
    if (selectedMetric) {
      fetchAnalyticsQuery(selectedMetric);
    }
  }, [selectedMetric]);

  const getRiskColor = (impact, likelihood) => {
    const riskScore = (impact + likelihood) / 2;
    if (riskScore >= 80) return '#dc3545'; // Critical
    if (riskScore >= 60) return '#fd7e14'; // High
    if (riskScore >= 40) return '#ffc107'; // Medium
    return '#28a745'; // Low
  };

  const getPriorityColor = (priority) => {
    const colors = {
      'Critical': '#dc3545',
      'High': '#fd7e14',
      'Medium': '#ffc107',
      'Low': '#28a745'
    };
    return colors[priority] || '#6c757d';
  };

  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric'
    });
  };

  const getSeverityBadgeClass = (severity) => {
    const classes = {
      low: 'badge severity-low',
      medium: 'badge severity-medium',
      high: 'badge severity-high',
      critical: 'badge severity-critical'
    };
    return classes[severity] || 'badge';
  };

  if (loading) {
    return <div className="loading">Loading risk analytics...</div>;
  }

  return (
    <div style={{ padding: '20px', backgroundColor: '#f8f9fa', minHeight: '100vh' }}>
      <h1 className="page-title" style={{ marginBottom: '30px', color: '#333' }}>EHS Risk & Recommendations Dashboard</h1>

      {/* Risk Alert Cards - Top Row */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '20px', marginBottom: '30px' }}>
        {Object.entries(mockRiskData.riskAlerts).map(([key, alert]) => (
          <div key={key} style={{
            backgroundColor: 'white',
            padding: '25px',
            borderRadius: '12px',
            boxShadow: '0 2px 10px rgba(0,0,0,0.1)',
            border: '1px solid #e9ecef',
            borderLeft: `5px solid ${alert.color}`,
            textAlign: 'center'
          }}>
            <div style={{ fontSize: '36px', fontWeight: 'bold', color: alert.color, marginBottom: '8px' }}>
              {alert.value}
            </div>
            <div style={{ fontSize: '16px', color: '#666', fontWeight: '500' }}>
              {alert.label} {key !== 'actions' ? 'Risks' : 'Required'}
            </div>
            <div style={{ fontSize: '12px', color: alert.color, marginTop: '5px', fontWeight: 'bold' }}>
              {key === 'critical' ? '⚠️ URGENT' : key === 'high' ? '🔶 HIGH' : key === 'medium' ? '🔸 MEDIUM' : '📋 PENDING'}
            </div>
          </div>
        ))}
      </div>

      {/* Water Consumption Risk Analysis */}
      <div style={{
        backgroundColor: 'white',
        padding: '25px',
        borderRadius: '12px',
        boxShadow: '0 2px 10px rgba(0,0,0,0.1)',
        border: '1px solid #e9ecef',
        marginBottom: '30px'
      }}>
        <h3 style={{ marginBottom: '20px', color: '#333', display: 'flex', alignItems: 'center' }}>
          💧 Water Consumption Risk Analysis
        </h3>
        
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '30px' }}>
          {/* Trend Chart */}
          <div>
            <h4 style={{ marginBottom: '15px', color: '#666' }}>Consumption Trend vs Target</h4>
            <div style={{ display: 'flex', alignItems: 'end', gap: '15px', height: '150px', padding: '10px 0' }}>
              {mockRiskData.waterConsumption.trend.map((item, index) => (
                <div key={index} style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                  <div style={{ display: 'flex', gap: '5px', marginBottom: '10px' }}>
                    <div
                      style={{
                        width: '20px',
                        height: `${(item.consumption / 120) * 120}px`,
                        backgroundColor: item.consumption > item.target ? '#dc3545' : '#28a745',
                        borderRadius: '4px',
                        position: 'relative'
                      }}
                    >
                      <div style={{
                        position: 'absolute',
                        top: `${120 - (item.target / 120) * 120}px`,
                        left: '-5px',
                        right: '-5px',
                        height: '2px',
                        backgroundColor: '#007bff',
                        zIndex: 1
                      }}></div>
                    </div>
                  </div>
                  <div style={{ fontSize: '12px', color: '#666' }}>{item.month}</div>
                  <div style={{ fontSize: '10px', color: item.consumption > item.target ? '#dc3545' : '#28a745' }}>
                    {item.consumption}k gal
                  </div>
                </div>
              ))}
            </div>
            <div style={{ display: 'flex', justifyContent: 'center', gap: '20px', marginTop: '10px', fontSize: '12px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
                <div style={{ width: '12px', height: '12px', backgroundColor: '#dc3545', borderRadius: '2px' }}></div>
                <span>Actual</span>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
                <div style={{ width: '12px', height: '2px', backgroundColor: '#007bff' }}></div>
                <span>Target</span>
              </div>
            </div>
          </div>

          {/* Risk and Recommendation */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '15px' }}>
            <div style={{
              padding: '15px',
              backgroundColor: '#fff5f5',
              border: '1px solid #fed7d7',
              borderLeft: '4px solid #dc3545',
              borderRadius: '6px'
            }}>
              <div style={{ fontWeight: 'bold', color: '#dc3545', marginBottom: '5px' }}>🚨 RISK IDENTIFIED</div>
              <div style={{ fontSize: '14px', color: '#666' }}>{mockRiskData.waterConsumption.riskText}</div>
            </div>
            
            <div style={{
              padding: '15px',
              backgroundColor: '#f0f9ff',
              border: '1px solid #bae6fd',
              borderLeft: '4px solid #007bff',
              borderRadius: '6px'
            }}>
              <div style={{ fontWeight: 'bold', color: '#007bff', marginBottom: '5px' }}>💡 RECOMMENDATION</div>
              <div style={{ fontSize: '14px', color: '#666' }}>{mockRiskData.waterConsumption.recommendationText}</div>
            </div>
          </div>
        </div>
      </div>

      {/* Electricity Consumption Risk Analysis */}
      <div style={{
        backgroundColor: 'white',
        padding: '25px',
        borderRadius: '12px',
        boxShadow: '0 2px 10px rgba(0,0,0,0.1)',
        border: '1px solid #e9ecef',
        marginBottom: '30px'
      }}>
        <h3 style={{ marginBottom: '20px', color: '#333', display: 'flex', alignItems: 'center' }}>
          ⚡ Electricity Consumption Risk Analysis
        </h3>
        
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '30px' }}>
          {/* Trend Chart */}
          <div>
            <h4 style={{ marginBottom: '15px', color: '#666' }}>Usage vs Peak Demand</h4>
            <div style={{ display: 'flex', alignItems: 'end', gap: '15px', height: '150px', padding: '10px 0' }}>
              {mockRiskData.electricityConsumption.trend.map((item, index) => (
                <div key={index} style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                  <div style={{ display: 'flex', gap: '3px', marginBottom: '10px' }}>
                    <div
                      style={{
                        width: '15px',
                        height: `${(item.consumption / 500) * 120}px`,
                        backgroundColor: '#007bff',
                        borderRadius: '2px'
                      }}
                    ></div>
                    <div
                      style={{
                        width: '15px',
                        height: `${(item.peakDemand / 500) * 120}px`,
                        backgroundColor: item.peakDemand > 440 ? '#dc3545' : '#ffc107',
                        borderRadius: '2px'
                      }}
                    ></div>
                  </div>
                  <div style={{ fontSize: '12px', color: '#666' }}>{item.month}</div>
                  <div style={{ fontSize: '10px', color: '#666' }}>
                    {item.consumption}/{item.peakDemand}
                  </div>
                </div>
              ))}
            </div>
            <div style={{ display: 'flex', justifyContent: 'center', gap: '20px', marginTop: '10px', fontSize: '12px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
                <div style={{ width: '12px', height: '12px', backgroundColor: '#007bff', borderRadius: '2px' }}></div>
                <span>Usage (kWh)</span>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
                <div style={{ width: '12px', height: '12px', backgroundColor: '#dc3545', borderRadius: '2px' }}></div>
                <span>Peak Demand</span>
              </div>
            </div>
          </div>

          {/* Risk and Recommendation */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '15px' }}>
            <div style={{
              padding: '15px',
              backgroundColor: '#fff5f5',
              border: '1px solid #fed7d7',
              borderLeft: '4px solid #dc3545',
              borderRadius: '6px'
            }}>
              <div style={{ fontWeight: 'bold', color: '#dc3545', marginBottom: '5px' }}>🚨 RISK IDENTIFIED</div>
              <div style={{ fontSize: '14px', color: '#666' }}>{mockRiskData.electricityConsumption.riskText}</div>
            </div>
            
            <div style={{
              padding: '15px',
              backgroundColor: '#f0f9ff',
              border: '1px solid #bae6fd',
              borderLeft: '4px solid #007bff',
              borderRadius: '6px'
            }}>
              <div style={{ fontWeight: 'bold', color: '#007bff', marginBottom: '5px' }}>💡 RECOMMENDATION</div>
              <div style={{ fontSize: '14px', color: '#666' }}>{mockRiskData.electricityConsumption.recommendationText}</div>
            </div>
          </div>
        </div>
      </div>

      {/* Waste Generation Risk Analysis */}
      <div style={{
        backgroundColor: 'white',
        padding: '25px',
        borderRadius: '12px',
        boxShadow: '0 2px 10px rgba(0,0,0,0.1)',
        border: '1px solid #e9ecef',
        marginBottom: '30px'
      }}>
        <h3 style={{ marginBottom: '20px', color: '#333', display: 'flex', alignItems: 'center' }}>
          🗑️ Waste Generation Risk Analysis
        </h3>
        
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '30px' }}>
          {/* Trend Chart */}
          <div>
            <h4 style={{ marginBottom: '15px', color: '#666' }}>Total vs Hazardous Waste</h4>
            <div style={{ display: 'flex', alignItems: 'end', gap: '15px', height: '150px', padding: '10px 0' }}>
              {mockRiskData.wasteGeneration.trend.map((item, index) => (
                <div key={index} style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                  <div style={{ display: 'flex', gap: '3px', marginBottom: '10px' }}>
                    <div
                      style={{
                        width: '15px',
                        height: `${(item.waste / 60) * 120}px`,
                        backgroundColor: '#28a745',
                        borderRadius: '2px'
                      }}
                    ></div>
                    <div
                      style={{
                        width: '15px',
                        height: `${(item.hazardous / 60) * 120}px`,
                        backgroundColor: item.hazardous > 10 ? '#dc3545' : '#ffc107',
                        borderRadius: '2px'
                      }}
                    ></div>
                  </div>
                  <div style={{ fontSize: '12px', color: '#666' }}>{item.month}</div>
                  <div style={{ fontSize: '10px', color: '#666' }}>
                    {item.waste}/{item.hazardous} tons
                  </div>
                </div>
              ))}
            </div>
            <div style={{ display: 'flex', justifyContent: 'center', gap: '20px', marginTop: '10px', fontSize: '12px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
                <div style={{ width: '12px', height: '12px', backgroundColor: '#28a745', borderRadius: '2px' }}></div>
                <span>Total Waste</span>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
                <div style={{ width: '12px', height: '12px', backgroundColor: '#dc3545', borderRadius: '2px' }}></div>
                <span>Hazardous</span>
              </div>
            </div>
          </div>

          {/* Risk and Recommendation */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '15px' }}>
            <div style={{
              padding: '15px',
              backgroundColor: '#fff5f5',
              border: '1px solid #fed7d7',
              borderLeft: '4px solid #dc3545',
              borderRadius: '6px'
            }}>
              <div style={{ fontWeight: 'bold', color: '#dc3545', marginBottom: '5px' }}>🚨 RISK IDENTIFIED</div>
              <div style={{ fontSize: '14px', color: '#666' }}>{mockRiskData.wasteGeneration.riskText}</div>
            </div>
            
            <div style={{
              padding: '15px',
              backgroundColor: '#f0f9ff',
              border: '1px solid #bae6fd',
              borderLeft: '4px solid #007bff',
              borderRadius: '6px'
            }}>
              <div style={{ fontWeight: 'bold', color: '#007bff', marginBottom: '5px' }}>💡 RECOMMENDATION</div>
              <div style={{ fontSize: '14px', color: '#666' }}>{mockRiskData.wasteGeneration.recommendationText}</div>
            </div>
          </div>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '30px', marginBottom: '30px' }}>
        {/* Risk Priority Matrix */}
        <div style={{
          backgroundColor: 'white',
          padding: '25px',
          borderRadius: '12px',
          boxShadow: '0 2px 10px rgba(0,0,0,0.1)',
          border: '1px solid #e9ecef'
        }}>
          <h3 style={{ marginBottom: '20px', color: '#333' }}>📊 Risk Priority Matrix</h3>
          
          <div style={{ position: 'relative', width: '100%', height: '250px', backgroundColor: '#f8f9fa', borderRadius: '8px' }}>
            {/* Matrix Grid */}
            <div style={{
              position: 'absolute',
              top: '10px',
              left: '10px',
              right: '10px',
              bottom: '30px',
              border: '2px solid #dee2e6',
              borderRadius: '4px'
            }}>
              {/* Quadrant Lines */}
              <div style={{
                position: 'absolute',
                top: '50%',
                left: '0',
                right: '0',
                height: '1px',
                backgroundColor: '#dee2e6'
              }}></div>
              <div style={{
                position: 'absolute',
                top: '0',
                bottom: '0',
                left: '50%',
                width: '1px',
                backgroundColor: '#dee2e6'
              }}></div>
              
              {/* Risk Items */}
              {mockRiskData.riskMatrix.map((risk, index) => (
                <div
                  key={index}
                  style={{
                    position: 'absolute',
                    left: `${risk.likelihood}%`,
                    bottom: `${risk.impact}%`,
                    width: '12px',
                    height: '12px',
                    backgroundColor: getRiskColor(risk.impact, risk.likelihood),
                    borderRadius: '50%',
                    border: '2px solid white',
                    cursor: 'pointer',
                    transform: 'translate(-50%, 50%)',
                    boxShadow: '0 2px 4px rgba(0,0,0,0.2)'
                  }}
                  title={`${risk.name} (Impact: ${risk.impact}%, Likelihood: ${risk.likelihood}%)`}
                ></div>
              ))}
            </div>
            
            {/* Labels */}
            <div style={{ position: 'absolute', bottom: '5px', left: '50%', transform: 'translateX(-50%)', fontSize: '12px', color: '#666' }}>
              Likelihood →
            </div>
            <div style={{ 
              position: 'absolute', 
              top: '50%', 
              left: '5px', 
              transform: 'translateY(-50%) rotate(-90deg)', 
              fontSize: '12px', 
              color: '#666',
              transformOrigin: 'center'
            }}>
              Impact ↑
            </div>
          </div>
          
          {/* Legend */}
          <div style={{ marginTop: '15px' }}>
            <div style={{ fontSize: '12px', fontWeight: 'bold', marginBottom: '10px' }}>Risk Items:</div>
            {mockRiskData.riskMatrix.map((risk, index) => (
              <div key={index} style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '5px', fontSize: '12px' }}>
                <div style={{
                  width: '8px',
                  height: '8px',
                  backgroundColor: getRiskColor(risk.impact, risk.likelihood),
                  borderRadius: '50%'
                }}></div>
                <span>{risk.name} ({risk.type})</span>
              </div>
            ))}
          </div>
        </div>

        {/* Action Items Timeline */}
        <div style={{
          backgroundColor: 'white',
          padding: '25px',
          borderRadius: '12px',
          boxShadow: '0 2px 10px rgba(0,0,0,0.1)',
          border: '1px solid #e9ecef'
        }}>
          <h3 style={{ marginBottom: '20px', color: '#333' }}>📅 Action Items Timeline</h3>
          
          <div style={{ maxHeight: '400px', overflowY: 'auto' }}>
            {mockRiskData.actionItems.map((item) => (
              <div key={item.id} style={{
                padding: '15px',
                marginBottom: '15px',
                border: '1px solid #e9ecef',
                borderLeft: `4px solid ${getPriorityColor(item.priority)}`,
                borderRadius: '6px',
                backgroundColor: '#fff'
              }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'start', marginBottom: '8px' }}>
                  <div style={{ fontWeight: 'bold', fontSize: '14px', color: '#333', flex: 1 }}>
                    {item.task}
                  </div>
                  <span style={{
                    padding: '2px 8px',
                    borderRadius: '12px',
                    fontSize: '11px',
                    fontWeight: 'bold',
                    color: 'white',
                    backgroundColor: getPriorityColor(item.priority)
                  }}>
                    {item.priority}
                  </span>
                </div>
                
                <div style={{ fontSize: '12px', color: '#666', marginBottom: '8px' }}>
                  📍 {item.facility} • 📅 Due: {formatDate(item.deadline)}
                </div>
                
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                  <div style={{ flex: 1 }}>
                    <div style={{
                      height: '6px',
                      backgroundColor: '#e9ecef',
                      borderRadius: '3px',
                      overflow: 'hidden'
                    }}>
                      <div style={{
                        height: '100%',
                        width: `${item.status}%`,
                        backgroundColor: item.status === 0 ? '#dc3545' : item.status < 50 ? '#ffc107' : '#28a745',
                        borderRadius: '3px',
                        transition: 'width 0.3s ease'
                      }}></div>
                    </div>
                  </div>
                  <span style={{ fontSize: '12px', fontWeight: 'bold', color: '#666' }}>
                    {item.status}%
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Original Dashboard Data - Maintained for System Integration */}
      {dashboardData && (
        <div style={{
          backgroundColor: 'white',
          padding: '25px',
          borderRadius: '12px',
          boxShadow: '0 2px 10px rgba(0,0,0,0.1)',
          border: '1px solid #e9ecef',
          marginBottom: '30px'
        }}>
          <h2 style={{ marginBottom: '20px', color: '#333' }}>System Integration Metrics</h2>
          <div className="stats-grid">
            <div className="stat-card">
              <span className="stat-number">{dashboardData.total_incidents}</span>
              <span className="stat-label">Total Incidents</span>
            </div>
            <div className="stat-card">
              <span className="stat-number">{dashboardData.open_incidents}</span>
              <span className="stat-label">Open Incidents</span>
            </div>
            <div className="stat-card">
              <span className="stat-number">{dashboardData.compliance_rate}%</span>
              <span className="stat-label">Compliance Rate</span>
            </div>
            <div className="stat-card">
              <span className="stat-number">{dashboardData.overdue_audits}</span>
              <span className="stat-label">Overdue Audits</span>
            </div>
          </div>
        </div>
      )}

      {/* Analytics Selection - Maintained for Functionality */}
      <div className="card">
        <h2 className="chart-title">Detailed Risk Analytics</h2>
        <div style={{ marginBottom: '20px' }}>
          <label className="form-label">Select Analytics View:</label>
          <select 
            className="form-control" 
            value={selectedMetric}
            onChange={(e) => setSelectedMetric(e.target.value)}
            style={{ maxWidth: '300px' }}
          >
            <option value="incident_trends">Incident Trends</option>
            <option value="compliance_status">Compliance Status</option>
            <option value="safety_performance">Safety Performance</option>
          </select>
        </div>

        {/* Analytics Data Display */}
        {analyticsData && (
          <div>
            <h3 style={{ marginBottom: '20px', color: '#333' }}>
              {analyticsData.metric_type.replace('_', ' ').toUpperCase()}
            </h3>
            
            {selectedMetric === 'incident_trends' && (
              <div>
                <h4>Monthly Incident Breakdown</h4>
                <div style={{ overflowX: 'auto' }}>
                  <table className="table">
                    <thead>
                      <tr>
                        <th>Month</th>
                        <th>Total Incidents</th>
                        <th>Low Severity</th>
                        <th>Medium Severity</th>
                        <th>High Severity</th>
                      </tr>
                    </thead>
                    <tbody>
                      {analyticsData.data.map((item, index) => (
                        <tr key={index}>
                          <td>{item.month}</td>
                          <td>{item.incidents}</td>
                          <td>{item.severity_breakdown.low}</td>
                          <td>{item.severity_breakdown.medium}</td>
                          <td>{item.severity_breakdown.high}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <div className="card" style={{ marginTop: '20px', backgroundColor: '#f8f9fa' }}>
                  <h5>Summary</h5>
                  <p><strong>Total Incidents:</strong> {analyticsData.summary.total_incidents}</p>
                  <p><strong>Average per Month:</strong> {analyticsData.summary.avg_per_month}</p>
                  <p><strong>Trend:</strong> {analyticsData.summary.trend}</p>
                </div>
              </div>
            )}

            {selectedMetric === 'compliance_status' && (
              <div>
                <h4>Compliance by Regulation Type</h4>
                <div style={{ overflowX: 'auto' }}>
                  <table className="table">
                    <thead>
                      <tr>
                        <th>Regulation</th>
                        <th>Compliant</th>
                        <th>Non-Compliant</th>
                        <th>Pending</th>
                      </tr>
                    </thead>
                    <tbody>
                      {analyticsData.data.map((item, index) => (
                        <tr key={index}>
                          <td>{item.regulation}</td>
                          <td><span className="badge status-compliant">{item.compliant}</span></td>
                          <td><span className="badge status-non-compliant">{item.non_compliant}</span></td>
                          <td><span className="badge status-pending">{item.pending}</span></td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <div className="card" style={{ marginTop: '20px', backgroundColor: '#f8f9fa' }}>
                  <h5>Summary</h5>
                  <p><strong>Overall Compliance Rate:</strong> {analyticsData.summary.overall_compliance_rate}%</p>
                  <p><strong>Total Regulations:</strong> {analyticsData.summary.total_regulations}</p>
                </div>
              </div>
            )}

            {selectedMetric === 'safety_performance' && (
              <div>
                <h4>Safety Performance Metrics</h4>
                <div className="stats-grid">
                  {analyticsData.data.map((item, index) => (
                    <div key={index} className="stat-card">
                      <span className="stat-number">{item.value}</span>
                      <span className="stat-label">{item.metric}</span>
                    </div>
                  ))}
                </div>
                <div className="card" style={{ marginTop: '20px', backgroundColor: '#f8f9fa' }}>
                  <h5>Performance Summary</h5>
                  <p><strong>Safety Score:</strong> {analyticsData.summary.safety_score}/100</p>
                  <p><strong>Industry Benchmark:</strong> {analyticsData.summary.industry_benchmark}/100</p>
                  <p><strong>Performance vs Benchmark:</strong> 
                    <span style={{ 
                      color: analyticsData.summary.safety_score > analyticsData.summary.industry_benchmark ? '#28a745' : '#dc3545',
                      fontWeight: 'bold',
                      marginLeft: '5px'
                    }}>
                      {analyticsData.summary.safety_score > analyticsData.summary.industry_benchmark ? 'Above' : 'Below'} Average
                    </span>
                  </p>
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Recent Incidents - Maintained for System Integration */}
      {dashboardData && dashboardData.recent_incidents && (
        <div className="card">
          <h2 className="chart-title">Recent Incidents</h2>
          <div style={{ overflowX: 'auto' }}>
            <table className="table">
              <thead>
                <tr>
                  <th>ID</th>
                  <th>Title</th>
                  <th>Severity</th>
                  <th>Location</th>
                  <th>Date</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {dashboardData.recent_incidents.map((incident) => (
                  <tr key={incident.id}>
                    <td>{incident.id}</td>
                    <td>{incident.title}</td>
                    <td>
                      <span className={getSeverityBadgeClass(incident.severity)}>
                        {incident.severity}
                      </span>
                    </td>
                    <td>{incident.location}</td>
                    <td>{formatDate(incident.incident_date)}</td>
                    <td>{incident.status}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
};

export default Analytics;