import React, { useState, useEffect } from 'react';
import axios from 'axios';

const AnalyticsHierarchical = () => {
  const [dashboardData, setDashboardData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [currentView, setCurrentView] = useState('site'); // 'site', 'category', 'subsection'
  const [selectedCategory, setSelectedCategory] = useState(null);
  const [selectedSubsection, setSelectedSubsection] = useState(null);
  const [breadcrumb, setBreadcrumb] = useState(['Site 1']);

  // Reuse mock data from original Analytics component
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
      recommendationText: "RECOMMENDATION: Review water pump efficiency at Facility Mumbai. Replace pumps #3 and #5 which are operating at 60% efficiency. Estimated savings: 2,500 gallons/month",
      goals: {
        primary: "Reduce water consumption by 15% annually",
        secondary: "Maintain efficiency targets across all facilities",
        tertiary: "Implement smart water monitoring systems"
      },
      facts: {
        currentUsage: "100k gallons/month (25% above target)",
        facilities: ["Mumbai (40%)", "Shanghai (30%)", "Berlin (20%)", "London (10%)"],
        trend: "Increasing 5% monthly over last quarter"
      },
      analysis: {
        rootCause: "Pump inefficiency and scheduling issues",
        impact: "Budget overrun of $25,000 annually",
        riskLevel: "High - threatens sustainability goals"
      }
    },
    electricityConsumption: {
      trend: [
        { month: 'May', consumption: 320, peakDemand: 380 },
        { month: 'Jun', consumption: 340, peakDemand: 400 },
        { month: 'Jul', consumption: 365, peakDemand: 420 },
        { month: 'Aug', consumption: 385, peakDemand: 460 }
      ],
      riskText: "RISK: Peak demand charges at Facility Shanghai exceeded budget by 40% in August due to inefficient HVAC scheduling",
      recommendationText: "RECOMMENDATION: Implement smart HVAC controls with occupancy sensors. Adjust production schedules to avoid peak hours (2-6 PM). Potential savings: $15,000/month",
      goals: {
        primary: "Reduce peak demand charges by 30%",
        secondary: "Implement smart HVAC scheduling",
        tertiary: "Achieve 20% overall energy reduction"
      },
      facts: {
        currentUsage: "385 kWh/month with 460 kWh peak demand",
        facilities: ["Shanghai (45%)", "Mumbai (25%)", "Berlin (20%)", "London (10%)"],
        trend: "Peak demand increasing 8% monthly"
      },
      analysis: {
        rootCause: "Inefficient HVAC scheduling and peak hour operations",
        impact: "Additional $60,000 in peak demand charges annually",
        riskLevel: "Critical - immediate action required"
      }
    },
    wasteGeneration: {
      trend: [
        { month: 'May', waste: 45, hazardous: 8 },
        { month: 'Jun', waste: 48, hazardous: 9 },
        { month: 'Jul', waste: 52, hazardous: 12 },
        { month: 'Aug', waste: 55, hazardous: 15 }
      ],
      riskText: "RISK: Hazardous waste disposal costs increased 35% due to improper segregation at Facility Berlin",
      recommendationText: "RECOMMENDATION: Conduct waste segregation training for all staff. Install color-coded bins at 15 key locations. Expected reduction: 20% in disposal costs",
      goals: {
        primary: "Reduce hazardous waste generation by 25%",
        secondary: "Improve waste segregation compliance to 95%",
        tertiary: "Implement waste-to-energy programs"
      },
      facts: {
        currentGeneration: "55 tons total, 15 tons hazardous",
        facilities: ["Berlin (40%)", "Shanghai (30%)", "Mumbai (20%)", "London (10%)"],
        trend: "Hazardous waste increasing 20% quarterly"
      },
      analysis: {
        rootCause: "Poor segregation practices and inadequate training",
        impact: "Additional $45,000 in disposal costs annually",
        riskLevel: "High - compliance and cost risk"
      }
    }
  };

  const categories = [
    {
      id: 'electricity',
      title: 'Electricity Consumption',
      icon: '⚡',
      color: '#007bff',
      description: 'Monitor and optimize electrical energy usage across all facilities',
      status: 'Critical',
      data: mockRiskData.electricityConsumption
    },
    {
      id: 'water',
      title: 'Water Consumption', 
      icon: '💧',
      color: '#17a2b8',
      description: 'Track water usage patterns and conservation efforts',
      status: 'High',
      data: mockRiskData.waterConsumption
    },
    {
      id: 'waste',
      title: 'Waste Generation',
      icon: '🗑️', 
      color: '#28a745',
      description: 'Manage waste production and disposal processes',
      status: 'High',
      data: mockRiskData.wasteGeneration
    }
  ];

  const subsections = ['goals', 'facts', 'analysis', 'recommendations'];

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
      setError('Failed to fetch analytics data. Using mock data.');
      console.error('Error fetching dashboard data:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleCategoryClick = (category) => {
    setSelectedCategory(category);
    setCurrentView('category');
    setBreadcrumb(['Site 1', category.title]);
  };

  const handleSubsectionClick = (subsection) => {
    setSelectedSubsection(subsection);
    setCurrentView('subsection');
    setBreadcrumb(['Site 1', selectedCategory.title, subsection.charAt(0).toUpperCase() + subsection.slice(1)]);
  };

  const handleBreadcrumbClick = (index) => {
    const newBreadcrumb = breadcrumb.slice(0, index + 1);
    setBreadcrumb(newBreadcrumb);
    
    if (index === 0) {
      setCurrentView('site');
      setSelectedCategory(null);
      setSelectedSubsection(null);
    } else if (index === 1) {
      setCurrentView('category');
      setSelectedSubsection(null);
    }
  };

  const getStatusColor = (status) => {
    const colors = {
      'Critical': '#dc3545',
      'High': '#fd7e14', 
      'Medium': '#ffc107',
      'Low': '#28a745'
    };
    return colors[status] || '#6c757d';
  };

  const renderTrendChart = (trendData, type) => {
    if (type === 'water') {
      return (
        <div style={{ display: 'flex', alignItems: 'end', gap: '15px', height: '150px', padding: '10px 0' }}>
          {trendData.map((item, index) => (
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
      );
    } else if (type === 'electricity') {
      return (
        <div style={{ display: 'flex', alignItems: 'end', gap: '15px', height: '150px', padding: '10px 0' }}>
          {trendData.map((item, index) => (
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
      );
    } else if (type === 'waste') {
      return (
        <div style={{ display: 'flex', alignItems: 'end', gap: '15px', height: '150px', padding: '10px 0' }}>
          {trendData.map((item, index) => (
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
      );
    }
  };

  const renderBreadcrumb = () => (
    <div style={{ 
      marginBottom: '30px', 
      padding: '15px', 
      backgroundColor: 'white', 
      borderRadius: '8px',
      boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
      border: '1px solid #e9ecef'
    }}>
      <nav style={{ fontSize: '14px' }}>
        {breadcrumb.map((item, index) => (
          <span key={index}>
            <button
              onClick={() => handleBreadcrumbClick(index)}
              style={{
                background: 'none',
                border: 'none',
                color: index === breadcrumb.length - 1 ? '#333' : '#007bff',
                cursor: index === breadcrumb.length - 1 ? 'default' : 'pointer',
                textDecoration: index === breadcrumb.length - 1 ? 'none' : 'underline',
                fontWeight: index === breadcrumb.length - 1 ? 'bold' : 'normal',
                fontSize: '14px',
                padding: '0'
              }}
            >
              {item}
            </button>
            {index < breadcrumb.length - 1 && <span style={{ margin: '0 10px', color: '#666' }}>›</span>}
          </span>
        ))}
      </nav>
    </div>
  );

  const renderSiteView = () => (
    <div>
      <h1 style={{ marginBottom: '20px', color: '#333', fontSize: '2.5rem' }}>Site 1 - EHS Analytics Dashboard</h1>
      
      {/* Risk Alert Summary */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '20px', marginBottom: '40px' }}>
        {Object.entries(mockRiskData.riskAlerts).map(([key, alert]) => (
          <div key={key} style={{
            backgroundColor: 'white',
            padding: '25px',
            borderRadius: '12px',
            boxShadow: '0 4px 6px rgba(0,0,0,0.1)',
            border: '1px solid #e9ecef',
            borderLeft: `5px solid ${alert.color}`,
            textAlign: 'center',
            transition: 'transform 0.2s ease, box-shadow 0.2s ease'
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

      {/* Category Cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(350px, 1fr))', gap: '30px' }}>
        {categories.map((category) => (
          <div
            key={category.id}
            onClick={() => handleCategoryClick(category)}
            style={{
              backgroundColor: 'white',
              padding: '30px',
              borderRadius: '16px',
              boxShadow: '0 6px 20px rgba(0,0,0,0.1)',
              border: '1px solid #e9ecef',
              cursor: 'pointer',
              transition: 'all 0.3s ease',
              transform: 'translateY(0)',
              position: 'relative',
              overflow: 'hidden'
            }}
            onMouseEnter={(e) => {
              e.target.style.transform = 'translateY(-5px)';
              e.target.style.boxShadow = '0 12px 30px rgba(0,0,0,0.15)';
            }}
            onMouseLeave={(e) => {
              e.target.style.transform = 'translateY(0)';
              e.target.style.boxShadow = '0 6px 20px rgba(0,0,0,0.1)';
            }}
          >
            <div style={{
              position: 'absolute',
              top: '0',
              left: '0',
              right: '0',
              height: '4px',
              background: `linear-gradient(90deg, ${category.color}, ${category.color}88)`
            }}></div>
            
            <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', marginBottom: '20px' }}>
              <div style={{ fontSize: '48px', opacity: 0.8 }}>{category.icon}</div>
              <span style={{
                padding: '4px 12px',
                borderRadius: '20px',
                fontSize: '12px',
                fontWeight: 'bold',
                color: 'white',
                backgroundColor: getStatusColor(category.status)
              }}>
                {category.status}
              </span>
            </div>
            
            <h3 style={{ 
              color: '#333', 
              marginBottom: '15px',
              fontSize: '1.5rem',
              fontWeight: '600'
            }}>
              {category.title}
            </h3>
            
            <p style={{ 
              color: '#666', 
              fontSize: '14px', 
              lineHeight: '1.5',
              marginBottom: '20px'
            }}>
              {category.description}
            </p>

            <div style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              color: category.color,
              fontWeight: '500',
              fontSize: '14px'
            }}>
              <span>View Details</span>
              <span style={{ fontSize: '16px' }}>→</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );

  const renderCategoryView = () => (
    <div>
      <div style={{ display: 'flex', alignItems: 'center', gap: '15px', marginBottom: '30px' }}>
        <div style={{ fontSize: '48px' }}>{selectedCategory.icon}</div>
        <div>
          <h1 style={{ color: '#333', marginBottom: '5px', fontSize: '2.5rem' }}>
            {selectedCategory.title}
          </h1>
          <p style={{ color: '#666', fontSize: '16px', margin: '0' }}>
            {selectedCategory.description}
          </p>
        </div>
      </div>

      {/* Subsection Navigation Cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '25px' }}>
        {subsections.map((subsection) => {
          const icons = {
            goals: '🎯',
            facts: '📊', 
            analysis: '🔍',
            recommendations: '💡'
          };
          
          const colors = {
            goals: '#28a745',
            facts: '#007bff',
            analysis: '#fd7e14', 
            recommendations: '#6f42c1'
          };

          return (
            <div
              key={subsection}
              onClick={() => handleSubsectionClick(subsection)}
              style={{
                backgroundColor: 'white',
                padding: '25px',
                borderRadius: '12px',
                boxShadow: '0 4px 15px rgba(0,0,0,0.1)',
                border: '1px solid #e9ecef',
                borderLeft: `5px solid ${colors[subsection]}`,
                cursor: 'pointer',
                transition: 'all 0.3s ease',
                position: 'relative'
              }}
              onMouseEnter={(e) => {
                e.target.style.transform = 'translateY(-3px)';
                e.target.style.boxShadow = '0 8px 25px rgba(0,0,0,0.15)';
              }}
              onMouseLeave={(e) => {
                e.target.style.transform = 'translateY(0)';
                e.target.style.boxShadow = '0 4px 15px rgba(0,0,0,0.1)';
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: '15px', marginBottom: '15px' }}>
                <div style={{ fontSize: '32px' }}>{icons[subsection]}</div>
                <h3 style={{ 
                  color: colors[subsection], 
                  margin: '0',
                  fontSize: '1.4rem',
                  fontWeight: '600',
                  textTransform: 'capitalize'
                }}>
                  {subsection}
                </h3>
              </div>
              
              <div style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                color: colors[subsection],
                fontWeight: '500',
                fontSize: '14px'
              }}>
                <span>View {subsection}</span>
                <span style={{ fontSize: '16px' }}>→</span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );

  const renderSubsectionView = () => {
    const data = selectedCategory.data;
    const subsection = selectedSubsection;
    
    const colors = {
      goals: '#28a745',
      facts: '#007bff', 
      analysis: '#fd7e14',
      recommendations: '#6f42c1'
    };

    const icons = {
      goals: '🎯',
      facts: '📊',
      analysis: '🔍', 
      recommendations: '💡'
    };

    return (
      <div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '15px', marginBottom: '30px' }}>
          <div style={{ fontSize: '48px' }}>{icons[subsection]}</div>
          <div>
            <h1 style={{ color: colors[subsection], marginBottom: '5px', fontSize: '2.5rem', textTransform: 'capitalize' }}>
              {subsection}
            </h1>
            <p style={{ color: '#666', fontSize: '16px', margin: '0' }}>
              {selectedCategory.title} - {subsection} overview and details
            </p>
          </div>
        </div>

        {/* Content based on subsection */}
        {subsection === 'goals' && (
          <div style={{ display: 'grid', gap: '25px' }}>
            {Object.entries(data.goals).map(([key, goal]) => (
              <div key={key} style={{
                backgroundColor: 'white',
                padding: '25px',
                borderRadius: '12px',
                boxShadow: '0 4px 15px rgba(0,0,0,0.1)',
                border: '1px solid #e9ecef',
                borderLeft: `5px solid ${colors.goals}`
              }}>
                <h3 style={{ color: colors.goals, marginBottom: '15px', textTransform: 'capitalize' }}>
                  {key} Goal
                </h3>
                <p style={{ color: '#333', fontSize: '16px', lineHeight: '1.6' }}>
                  {goal}
                </p>
              </div>
            ))}
          </div>
        )}

        {subsection === 'facts' && (
          <div style={{ display: 'grid', gap: '25px' }}>
            <div style={{
              backgroundColor: 'white',
              padding: '25px',
              borderRadius: '12px',
              boxShadow: '0 4px 15px rgba(0,0,0,0.1)',
              border: '1px solid #e9ecef',
              borderLeft: `5px solid ${colors.facts}`
            }}>
              <h3 style={{ color: colors.facts, marginBottom: '20px' }}>Current Usage</h3>
              <p style={{ color: '#333', fontSize: '18px', fontWeight: '600' }}>
                {data.facts.currentUsage}
              </p>
            </div>

            <div style={{
              backgroundColor: 'white',
              padding: '25px', 
              borderRadius: '12px',
              boxShadow: '0 4px 15px rgba(0,0,0,0.1)',
              border: '1px solid #e9ecef',
              borderLeft: `5px solid ${colors.facts}`
            }}>
              <h3 style={{ color: colors.facts, marginBottom: '20px' }}>Facility Breakdown</h3>
              <ul style={{ listStyle: 'none', padding: '0' }}>
                {data.facts.facilities.map((facility, index) => (
                  <li key={index} style={{ 
                    padding: '10px 0', 
                    borderBottom: index < data.facts.facilities.length - 1 ? '1px solid #e9ecef' : 'none',
                    fontSize: '16px',
                    color: '#333'
                  }}>
                    {facility}
                  </li>
                ))}
              </ul>
            </div>

            <div style={{
              backgroundColor: 'white',
              padding: '25px',
              borderRadius: '12px', 
              boxShadow: '0 4px 15px rgba(0,0,0,0.1)',
              border: '1px solid #e9ecef',
              borderLeft: `5px solid ${colors.facts}`
            }}>
              <h3 style={{ color: colors.facts, marginBottom: '15px' }}>Trend Analysis</h3>
              <p style={{ color: '#333', fontSize: '16px' }}>
                {data.facts.trend}
              </p>
              
              {/* Include trend chart */}
              <div style={{ marginTop: '20px' }}>
                <h4 style={{ marginBottom: '15px', color: '#666' }}>Usage Trend</h4>
                {renderTrendChart(data.trend, selectedCategory.id)}
              </div>
            </div>
          </div>
        )}

        {subsection === 'analysis' && (
          <div style={{ display: 'grid', gap: '25px' }}>
            <div style={{
              backgroundColor: 'white',
              padding: '25px',
              borderRadius: '12px',
              boxShadow: '0 4px 15px rgba(0,0,0,0.1)',
              border: '1px solid #e9ecef',
              borderLeft: `5px solid ${colors.analysis}`
            }}>
              <h3 style={{ color: colors.analysis, marginBottom: '20px' }}>Root Cause Analysis</h3>
              <p style={{ color: '#333', fontSize: '16px', lineHeight: '1.6' }}>
                {data.analysis.rootCause}
              </p>
            </div>

            <div style={{
              backgroundColor: 'white',
              padding: '25px',
              borderRadius: '12px',
              boxShadow: '0 4px 15px rgba(0,0,0,0.1)', 
              border: '1px solid #e9ecef',
              borderLeft: `5px solid ${colors.analysis}`
            }}>
              <h3 style={{ color: colors.analysis, marginBottom: '20px' }}>Financial Impact</h3>
              <p style={{ color: '#333', fontSize: '16px', lineHeight: '1.6' }}>
                {data.analysis.impact}
              </p>
            </div>

            <div style={{
              backgroundColor: 'white',
              padding: '25px',
              borderRadius: '12px',
              boxShadow: '0 4px 15px rgba(0,0,0,0.1)',
              border: '1px solid #e9ecef',
              borderLeft: `5px solid ${colors.analysis}`
            }}>
              <h3 style={{ color: colors.analysis, marginBottom: '20px' }}>Risk Assessment</h3>
              <div style={{
                padding: '15px',
                backgroundColor: '#fff3cd',
                border: '1px solid #ffeaa7',
                borderRadius: '6px',
                borderLeft: '4px solid #fd7e14'
              }}>
                <p style={{ color: '#333', fontSize: '16px', lineHeight: '1.6', margin: '0' }}>
                  {data.analysis.riskLevel}
                </p>
              </div>
            </div>

            {/* Risk identification from original data */}
            <div style={{
              backgroundColor: 'white',
              padding: '25px',
              borderRadius: '12px',
              boxShadow: '0 4px 15px rgba(0,0,0,0.1)',
              border: '1px solid #e9ecef',
              borderLeft: `5px solid #dc3545`
            }}>
              <h3 style={{ color: '#dc3545', marginBottom: '20px' }}>🚨 Risk Identified</h3>
              <p style={{ color: '#333', fontSize: '16px', lineHeight: '1.6' }}>
                {data.riskText}
              </p>
            </div>
          </div>
        )}

        {subsection === 'recommendations' && (
          <div style={{
            backgroundColor: 'white',
            padding: '30px',
            borderRadius: '12px',
            boxShadow: '0 4px 15px rgba(0,0,0,0.1)',
            border: '1px solid #e9ecef',
            borderLeft: `5px solid ${colors.recommendations}`
          }}>
            <h3 style={{ color: colors.recommendations, marginBottom: '25px', fontSize: '1.5rem' }}>
              💡 Recommended Actions
            </h3>
            <div style={{
              padding: '20px',
              backgroundColor: '#f8f9ff',
              border: '1px solid #e3e7ff', 
              borderRadius: '8px',
              borderLeft: '4px solid #6f42c1'
            }}>
              <p style={{ color: '#333', fontSize: '16px', lineHeight: '1.8', margin: '0' }}>
                {data.recommendationText}
              </p>
            </div>
            
            <div style={{ marginTop: '25px', padding: '20px', backgroundColor: '#f0f9ff', borderRadius: '8px' }}>
              <h4 style={{ color: colors.recommendations, marginBottom: '15px' }}>Implementation Priority</h4>
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                <span style={{
                  padding: '4px 12px',
                  backgroundColor: getStatusColor(selectedCategory.status),
                  color: 'white',
                  borderRadius: '20px',
                  fontSize: '12px',
                  fontWeight: 'bold'
                }}>
                  {selectedCategory.status} Priority
                </span>
                <span style={{ color: '#666', fontSize: '14px' }}>
                  Immediate action required for optimal results
                </span>
              </div>
            </div>
          </div>
        )}
      </div>
    );
  };

  if (loading) {
    return (
      <div style={{ 
        display: 'flex', 
        justifyContent: 'center', 
        alignItems: 'center', 
        minHeight: '100vh',
        fontSize: '18px',
        color: '#666'
      }}>
        Loading hierarchical analytics...
      </div>
    );
  }

  return (
    <div style={{ 
      padding: '30px', 
      backgroundColor: '#f8f9fa', 
      minHeight: '100vh',
      transition: 'all 0.3s ease'
    }}>
      {renderBreadcrumb()}
      
      <div style={{ transition: 'all 0.3s ease' }}>
        {currentView === 'site' && renderSiteView()}
        {currentView === 'category' && renderCategoryView()}  
        {currentView === 'subsection' && renderSubsectionView()}
      </div>
    </div>
  );
};

export default AnalyticsHierarchical;