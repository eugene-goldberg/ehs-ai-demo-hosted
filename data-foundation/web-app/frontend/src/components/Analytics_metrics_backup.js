import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { API_ENDPOINTS } from '../config/api';

const Analytics = () => {
  const [dashboardData, setDashboardData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedMetric, setSelectedMetric] = useState('incident_trends');
  const [analyticsData, setAnalyticsData] = useState(null);
  const [dateRange, setDateRange] = useState('Aug 25 - Sep 25');

  // Mock data for EHS dashboard
  const mockEHSData = {
    topMetrics: {
      totalEmissionsTracked: { value: '404K', change: '+3.47%', period: 'This week' },
      activeFacilities: { value: '300', change: '+2.14%', period: 'This week' },
      complianceScore: { value: '94%', change: '+1.42%', period: 'This week' },
      documentsProcessed: { value: '140K', change: '+6.70%', period: 'This week' }
    },
    trafficData: {
      campaign: 80,
      direct: 20,
      photoClicks: 70,
      linkClicks: 30
    },
    utilityTrends: [
      { day: 'Mon', electricity: 325, water: 280 },
      { day: 'Tue', electricity: 310, water: 295 },
      { day: 'Wed', electricity: 340, water: 270 },
      { day: 'Thu', electricity: 355, water: 285 },
      { day: 'Fri', electricity: 325, water: 300 },
      { day: 'Sat', electricity: 280, water: 250 },
      { day: 'Sun', electricity: 260, water: 240 }
    ],
    facilityLocations: [
      { country: 'India', facilities: 45, color: '#28a745' },
      { country: 'China', facilities: 38, color: '#ffc107' },
      { country: 'Russia', facilities: 22, color: '#dc3545' },
      { country: 'USA', facilities: 35, color: '#007bff' }
    ],
    complianceMetrics: {
      videoLectures: { completed: 12, total: 15 },
      safetyTraining: { completed: 14, total: 15 },
      auditCompliance: { completed: 11, total: 15 }
    }
  };

  useEffect(() => {
    fetchDashboardData();
  }, []);

  const fetchDashboardData = async () => {
    try {
      setLoading(true);
      const response = await axios.get(API_ENDPOINTS.analyticsDashboard);
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
      const response = await axios.post(API_ENDPOINTS.analyticsQuery, {
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

  const renderCircularProgress = (completed, total, label) => {
    const percentage = (completed / total) * 100;
    const strokeDasharray = `${percentage * 2.83} 283`;
    
    return (
      <div style={{ textAlign: 'center', margin: '10px' }}>
        <div style={{ position: 'relative', width: '80px', height: '80px', margin: '0 auto' }}>
          <svg width="80" height="80" style={{ transform: 'rotate(-90deg)' }}>
            <circle
              cx="40"
              cy="40"
              r="35"
              stroke="#e6e6e6"
              strokeWidth="6"
              fill="transparent"
            />
            <circle
              cx="40"
              cy="40"
              r="35"
              stroke="#007bff"
              strokeWidth="6"
              fill="transparent"
              strokeDasharray={strokeDasharray}
              strokeLinecap="round"
            />
          </svg>
          <div style={{
            position: 'absolute',
            top: '50%',
            left: '50%',
            transform: 'translate(-50%, -50%)',
            fontSize: '12px',
            fontWeight: 'bold'
          }}>
            {completed}/{total}
          </div>
        </div>
        <div style={{ fontSize: '12px', marginTop: '5px', color: '#666' }}>{label}</div>
      </div>
    );
  };

  if (loading) {
    return <div className="loading">Loading analytics...</div>;
  }

  return (
    <div style={{ padding: '20px', backgroundColor: '#f8f9fa', minHeight: '100vh' }}>
      <h1 className="page-title" style={{ marginBottom: '30px', color: '#333' }}>EHS Analytics Dashboard</h1>

      {/* Top Metrics Row */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '20px', marginBottom: '30px' }}>
        <div style={{
          backgroundColor: 'white',
          padding: '25px',
          borderRadius: '12px',
          boxShadow: '0 2px 10px rgba(0,0,0,0.1)',
          border: '1px solid #e9ecef'
        }}>
          <div style={{ fontSize: '32px', fontWeight: 'bold', color: '#28a745', marginBottom: '8px' }}>
            {mockEHSData.topMetrics.totalEmissionsTracked.value}
          </div>
          <div style={{ fontSize: '16px', color: '#666', marginBottom: '5px' }}>Total Emissions Tracked</div>
          <div style={{ fontSize: '14px', color: '#28a745' }}>
            {mockEHSData.topMetrics.totalEmissionsTracked.change} ({mockEHSData.topMetrics.totalEmissionsTracked.period})
          </div>
        </div>

        <div style={{
          backgroundColor: 'white',
          padding: '25px',
          borderRadius: '12px',
          boxShadow: '0 2px 10px rgba(0,0,0,0.1)',
          border: '1px solid #e9ecef'
        }}>
          <div style={{ fontSize: '32px', fontWeight: 'bold', color: '#007bff', marginBottom: '8px' }}>
            {mockEHSData.topMetrics.activeFacilities.value}
          </div>
          <div style={{ fontSize: '16px', color: '#666', marginBottom: '5px' }}>Active Facilities</div>
          <div style={{ fontSize: '14px', color: '#28a745' }}>
            {mockEHSData.topMetrics.activeFacilities.change} ({mockEHSData.topMetrics.activeFacilities.period})
          </div>
        </div>

        <div style={{
          backgroundColor: 'white',
          padding: '25px',
          borderRadius: '12px',
          boxShadow: '0 2px 10px rgba(0,0,0,0.1)',
          border: '1px solid #e9ecef'
        }}>
          <div style={{ fontSize: '32px', fontWeight: 'bold', color: '#ffc107', marginBottom: '8px' }}>
            {mockEHSData.topMetrics.complianceScore.value}
          </div>
          <div style={{ fontSize: '16px', color: '#666', marginBottom: '5px' }}>Compliance Score</div>
          <div style={{ fontSize: '14px', color: '#28a745' }}>
            {mockEHSData.topMetrics.complianceScore.change} ({mockEHSData.topMetrics.complianceScore.period})
          </div>
        </div>

        <div style={{
          backgroundColor: 'white',
          padding: '25px',
          borderRadius: '12px',
          boxShadow: '0 2px 10px rgba(0,0,0,0.1)',
          border: '1px solid #e9ecef'
        }}>
          <div style={{ fontSize: '32px', fontWeight: 'bold', color: '#dc3545', marginBottom: '8px' }}>
            {mockEHSData.topMetrics.documentsProcessed.value}
          </div>
          <div style={{ fontSize: '16px', color: '#666', marginBottom: '5px' }}>Documents Processed</div>
          <div style={{ fontSize: '14px', color: '#28a745' }}>
            {mockEHSData.topMetrics.documentsProcessed.change} ({mockEHSData.topMetrics.documentsProcessed.period})
          </div>
        </div>
      </div>

      {/* Visitors Overview Section */}
      <div style={{
        backgroundColor: 'white',
        padding: '25px',
        borderRadius: '12px',
        boxShadow: '0 2px 10px rgba(0,0,0,0.1)',
        border: '1px solid #e9ecef',
        marginBottom: '30px'
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
          <h3 style={{ margin: 0, color: '#333' }}>Data Sources Overview</h3>
          <select 
            value={dateRange}
            onChange={(e) => setDateRange(e.target.value)}
            style={{
              padding: '8px 12px',
              border: '1px solid #ddd',
              borderRadius: '6px',
              backgroundColor: 'white'
            }}
          >
            <option value="Aug 25 - Sep 25">Aug 25 - Sep 25</option>
            <option value="Sep 25 - Oct 25">Sep 25 - Oct 25</option>
            <option value="Oct 25 - Nov 25">Oct 25 - Nov 25</option>
          </select>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: '30px' }}>
          <div style={{ display: 'flex', gap: '20px' }}>
            {/* Automated Data Sources */}
            <div style={{ flex: 1 }}>
              <div style={{ textAlign: 'center', marginBottom: '15px' }}>
                <div style={{ fontSize: '18px', fontWeight: 'bold', color: '#333' }}>Automated Sources</div>
                <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#007bff' }}>
                  {mockEHSData.trafficData.campaign}%
                </div>
              </div>
              <div style={{
                height: '150px',
                backgroundColor: '#007bff',
                borderRadius: '8px',
                display: 'flex',
                alignItems: 'end',
                justifyContent: 'center',
                color: 'white',
                fontWeight: 'bold'
              }}>
                <div style={{ padding: '10px' }}>Sensor Data, IoT Devices</div>
              </div>
            </div>

            {/* Manual Data Entry */}
            <div style={{ flex: 1 }}>
              <div style={{ textAlign: 'center', marginBottom: '15px' }}>
                <div style={{ fontSize: '18px', fontWeight: 'bold', color: '#333' }}>Manual Entry</div>
                <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#6c757d' }}>
                  {mockEHSData.trafficData.direct}%
                </div>
              </div>
              <div style={{
                height: '150px',
                backgroundColor: '#6c757d',
                borderRadius: '8px',
                display: 'flex',
                alignItems: 'end',
                justifyContent: 'center',
                color: 'white',
                fontWeight: 'bold'
              }}>
                <div style={{ padding: '10px' }}>Staff Reports, Inspections</div>
              </div>
            </div>
          </div>

          {/* Data Interaction Metrics */}
          <div>
            <h4 style={{ marginBottom: '20px', color: '#333' }}>Data Interaction</h4>
            <div style={{ marginBottom: '15px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '5px' }}>
                <span>Report Views</span>
                <span style={{ fontWeight: 'bold' }}>{mockEHSData.trafficData.photoClicks}%</span>
              </div>
              <div style={{
                height: '8px',
                backgroundColor: '#e9ecef',
                borderRadius: '4px',
                overflow: 'hidden'
              }}>
                <div style={{
                  height: '100%',
                  width: `${mockEHSData.trafficData.photoClicks}%`,
                  backgroundColor: '#28a745',
                  borderRadius: '4px'
                }}></div>
              </div>
            </div>
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '5px' }}>
                <span>Document Downloads</span>
                <span style={{ fontWeight: 'bold' }}>{mockEHSData.trafficData.linkClicks}%</span>
              </div>
              <div style={{
                height: '8px',
                backgroundColor: '#e9ecef',
                borderRadius: '4px',
                overflow: 'hidden'
              }}>
                <div style={{
                  height: '100%',
                  width: `${mockEHSData.trafficData.linkClicks}%`,
                  backgroundColor: '#ffc107',
                  borderRadius: '4px'
                }}></div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Utility Consumption Trends */}
      <div style={{
        backgroundColor: 'white',
        padding: '25px',
        borderRadius: '12px',
        boxShadow: '0 2px 10px rgba(0,0,0,0.1)',
        border: '1px solid #e9ecef',
        marginBottom: '30px'
      }}>
        <h3 style={{ marginBottom: '20px', color: '#333' }}>Utility Consumption Trends</h3>
        <div style={{ display: 'flex', alignItems: 'end', gap: '15px', height: '200px', padding: '20px 0' }}>
          {mockEHSData.utilityTrends.map((item, index) => (
            <div key={index} style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
              <div style={{ display: 'flex', gap: '3px', marginBottom: '10px' }}>
                <div
                  style={{
                    width: '15px',
                    height: `${(item.electricity / 400) * 150}px`,
                    backgroundColor: '#007bff',
                    borderRadius: '2px'
                  }}
                  title={`Electricity: ${item.electricity} kWh`}
                ></div>
                <div
                  style={{
                    width: '15px',
                    height: `${(item.water / 400) * 150}px`,
                    backgroundColor: '#28a745',
                    borderRadius: '2px'
                  }}
                  title={`Water: ${item.water} gallons`}
                ></div>
              </div>
              <div style={{ fontSize: '12px', color: '#666' }}>{item.day}</div>
            </div>
          ))}
        </div>
        <div style={{ display: 'flex', justifyContent: 'center', gap: '20px', marginTop: '15px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
            <div style={{ width: '15px', height: '15px', backgroundColor: '#007bff', borderRadius: '2px' }}></div>
            <span style={{ fontSize: '14px' }}>Electricity (kWh)</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
            <div style={{ width: '15px', height: '15px', backgroundColor: '#28a745', borderRadius: '2px' }}></div>
            <span style={{ fontSize: '14px' }}>Water (gallons)</span>
          </div>
        </div>
        <div style={{ textAlign: 'center', marginTop: '15px', color: '#666' }}>
          Average: 325 kWh electricity, 280 gallons water
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: '30px', marginBottom: '30px' }}>
        {/* Facility Locations Map */}
        <div style={{
          backgroundColor: 'white',
          padding: '25px',
          borderRadius: '12px',
          boxShadow: '0 2px 10px rgba(0,0,0,0.1)',
          border: '1px solid #e9ecef'
        }}>
          <h3 style={{ marginBottom: '20px', color: '#333' }}>Global Facility Distribution</h3>
          
          {/* Simplified World Map Representation */}
          <div style={{
            height: '200px',
            backgroundColor: '#f8f9fa',
            borderRadius: '8px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            marginBottom: '20px',
            border: '2px dashed #dee2e6',
            position: 'relative'
          }}>
            <div style={{ textAlign: 'center', color: '#6c757d' }}>
              <div style={{ fontSize: '48px', marginBottom: '10px' }}>🗺️</div>
              <div>Global Facility Map</div>
            </div>
            
            {/* Facility Markers */}
            <div style={{ position: 'absolute', top: '30%', left: '20%', width: '10px', height: '10px', backgroundColor: '#28a745', borderRadius: '50%', border: '2px solid white' }}></div>
            <div style={{ position: 'absolute', top: '25%', left: '60%', width: '10px', height: '10px', backgroundColor: '#ffc107', borderRadius: '50%', border: '2px solid white' }}></div>
            <div style={{ position: 'absolute', top: '20%', left: '70%', width: '10px', height: '10px', backgroundColor: '#dc3545', borderRadius: '50%', border: '2px solid white' }}></div>
            <div style={{ position: 'absolute', top: '35%', left: '15%', width: '10px', height: '10px', backgroundColor: '#007bff', borderRadius: '50%', border: '2px solid white' }}></div>
          </div>

          {/* Facility Distribution Chart */}
          <div>
            {mockEHSData.facilityLocations.map((location, index) => (
              <div key={index} style={{ display: 'flex', alignItems: 'center', marginBottom: '10px' }}>
                <div style={{ 
                  width: '80px', 
                  fontSize: '14px', 
                  fontWeight: 'bold',
                  color: '#333'
                }}>
                  {location.country}
                </div>
                <div style={{ flex: 1, marginRight: '10px' }}>
                  <div style={{
                    height: '20px',
                    backgroundColor: '#e9ecef',
                    borderRadius: '10px',
                    overflow: 'hidden'
                  }}>
                    <div style={{
                      height: '100%',
                      width: `${(location.facilities / 50) * 100}%`,
                      backgroundColor: location.color,
                      borderRadius: '10px'
                    }}></div>
                  </div>
                </div>
                <div style={{ fontSize: '14px', fontWeight: 'bold', color: '#333', width: '30px' }}>
                  {location.facilities}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Compliance Summary */}
        <div style={{
          backgroundColor: 'white',
          padding: '25px',
          borderRadius: '12px',
          boxShadow: '0 2px 10px rgba(0,0,0,0.1)',
          border: '1px solid #e9ecef'
        }}>
          <h3 style={{ marginBottom: '20px', color: '#333' }}>Compliance Summary</h3>
          
          <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
            {renderCircularProgress(
              mockEHSData.complianceMetrics.videoLectures.completed,
              mockEHSData.complianceMetrics.videoLectures.total,
              'Safety Training'
            )}
            {renderCircularProgress(
              mockEHSData.complianceMetrics.safetyTraining.completed,
              mockEHSData.complianceMetrics.safetyTraining.total,
              'Audit Completion'
            )}
            {renderCircularProgress(
              mockEHSData.complianceMetrics.auditCompliance.completed,
              mockEHSData.complianceMetrics.auditCompliance.total,
              'Certification Status'
            )}
          </div>
        </div>
      </div>

      {/* Original Dashboard Data */}
      {dashboardData && (
        <div style={{
          backgroundColor: 'white',
          padding: '25px',
          borderRadius: '12px',
          boxShadow: '0 2px 10px rgba(0,0,0,0.1)',
          border: '1px solid #e9ecef',
          marginBottom: '30px'
        }}>
          <h2 style={{ marginBottom: '20px', color: '#333' }}>System Metrics</h2>
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

      {/* Analytics Selection */}
      <div className="card">
        <h2 className="chart-title">Detailed Analytics</h2>
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

      {/* Recent Incidents */}
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