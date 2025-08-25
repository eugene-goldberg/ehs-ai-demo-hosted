import React, { useState, useEffect } from 'react';
import axios from 'axios';
import IngestionProgress from './IngestionProgress';
import { API_ENDPOINTS } from '../config/api';

const DataManagement = () => {
  const [processedDocuments, setProcessedDocuments] = useState([]);
  const [rejectedDocuments, setRejectedDocuments] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [isIngesting, setIsIngesting] = useState(false);
  const [downloadingFiles, setDownloadingFiles] = useState(new Set());
  
  // Popup state
  const [hoveredRow, setHoveredRow] = useState(null);
  const [popupPosition, setPopupPosition] = useState({ x: 0, y: 0 });
  const [documentDetails, setDocumentDetails] = useState({});
  const [loadingDetails, setLoadingDetails] = useState({});

  useEffect(() => {
    fetchData();
  }, []);

  const fetchData = async () => {
    try {
      setLoading(true);
      const [documentsRes, rejectedRes] = await Promise.all([
        axios.get('http://localhost:8001/api/data/processed-documents'),
        axios.get(API_ENDPOINTS.rejectedDocuments)
      ]);
      
      // Extract documents array from API response or default to empty array
      const processedDocs = Array.isArray(documentsRes.data) 
        ? documentsRes.data 
        : (documentsRes.data.documents || []);
      
      const rejectedDocs = Array.isArray(rejectedRes.data) 
        ? rejectedRes.data 
        : (rejectedRes.data.documents || []);
      
      setProcessedDocuments(processedDocs);
      setRejectedDocuments(rejectedDocs);
      setError(null);
    } catch (err) {
      setError('Failed to fetch data. Please try again.');
      console.error('Error fetching data:', err);
      // Ensure arrays are initialized even on error
      setProcessedDocuments([]);
      setRejectedDocuments([]);
    } finally {
      setLoading(false);
    }
  };

  const fetchDocumentDetails = async (documentId) => {
    if (documentDetails[documentId] || loadingDetails[documentId]) {
      return; // Already loaded or loading
    }

    try {
      setLoadingDetails(prev => ({ ...prev, [documentId]: true }));
      const response = await axios.get(`http://localhost:8001/api/data/processed-documents/${documentId}`);
      setDocumentDetails(prev => ({ ...prev, [documentId]: response.data }));
    } catch (err) {
      console.error('Error fetching document details:', err);
      // Set empty details to prevent infinite loading
      setDocumentDetails(prev => ({ ...prev, [documentId]: null }));
    } finally {
      setLoadingDetails(prev => ({ ...prev, [documentId]: false }));
    }
  };

  const handleDownload = async (doc) => {
    if (!doc.file_path || downloadingFiles.has(doc.id)) {
      return;
    }

    try {
      setDownloadingFiles(prev => new Set(prev).add(doc.id));
      
      const response = await axios.get(`http://localhost:8001/api/data/documents/${doc.id}/download`, {
        responseType: 'blob',
        timeout: 30000 // 30 second timeout for large files
      });

      // Extract filename from response headers or use a default
      const contentDisposition = response.headers['content-disposition'];
      let filename = 'document.pdf'; // Default with .pdf extension
      
      if (contentDisposition) {
        const filenameMatch = contentDisposition.match(/filename[*]?=['"]?([^;\r\n"]+)['"]?/);
        if (filenameMatch) {
          filename = filenameMatch[1];
        }
      } else if (doc.file_path) {
        // Extract filename from file_path
        const pathParts = doc.file_path.split('/');
        filename = pathParts[pathParts.length - 1];
      } else if (doc.original_filename || doc.filename) {
        filename = doc.original_filename || doc.filename;
      }

      // Create blob URL and trigger download
      const blob = new Blob([response.data]);
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      
      // Cleanup
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);
      
    } catch (err) {
      console.error('Error downloading file:', err);
      
      // Show user-friendly error message
      let errorMessage = 'Failed to download file. ';
      if (err.response?.status === 404) {
        errorMessage += 'File not found on server.';
      } else if (err.code === 'ECONNABORTED') {
        errorMessage += 'Download timeout - file may be too large.';
      } else if (err.response?.status >= 500) {
        errorMessage += 'Server error occurred.';
      } else {
        errorMessage += 'Please try again.';
      }
      
      // You could show this in a toast notification or alert
      alert(errorMessage);
    } finally {
      setDownloadingFiles(prev => {
        const newSet = new Set(prev);
        newSet.delete(doc.id);
        return newSet;
      });
    }
  };

  const handleBulkIngestion = async () => {
    setIsIngesting(true);
    try {
      const response = await axios.post('http://localhost:8000/api/v1/ingest/batch', {
        clear_database: true
      });
      
      if (response.data.status === 'success') {
        console.log(`Ingestion completed successfully! ${response.data.data.successful_ingestions}/3 documents processed.`);
        // Refresh the data after ingestion
        fetchData();
        // Clear cached details
        setDocumentDetails({});
      } else {
        console.error('Ingestion failed:', response.data);
      }
    } catch (err) {
      console.error('Error calling bulk ingestion:', err);
    } finally {
      setIsIngesting(false);
    }
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

  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric'
    });
  };

  const formatCurrency = (amount) => {
    if (typeof amount !== 'number') return 'N/A';
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD'
    }).format(amount);
  };

  const formatNumber = (number, unit = '') => {
    if (typeof number !== 'number') return 'N/A';
    return new Intl.NumberFormat('en-US', {
      maximumFractionDigits: 2,
      minimumFractionDigits: 0
    }).format(number) + (unit ? ` ${unit}` : '');
  };

  const handleRowMouseEnter = async (event, docIndex) => {
    const rect = event.currentTarget.getBoundingClientRect();
    setPopupPosition({
      x: event.clientX + 10,
      y: event.clientY - 10
    });
    setHoveredRow(docIndex);

    // Fetch detailed document information
    const document = processedDocuments[docIndex];
    if (document && document.id) {
      await fetchDocumentDetails(document.id);
    }
  };

  const handleRowMouseMove = (event) => {
    setPopupPosition({
      x: event.clientX + 10,
      y: event.clientY - 10
    });
  };

  const handleRowMouseLeave = () => {
    setHoveredRow(null);
  };

  // Popup Component
  const DocumentPopup = ({ document, position }) => {
    if (!document) return null;

    const details = documentDetails[document.id];
    const isLoadingDetails = loadingDetails[document.id];

    const popupStyle = {
      position: 'fixed',
      left: `${position.x}px`,
      top: `${position.y}px`,
      backgroundColor: '#ffffff',
      border: 'none',
      borderRadius: '8px',
      padding: '16px',
      boxShadow: '0 8px 24px rgba(0, 0, 0, 0.15)',
      zIndex: 1000,
      minWidth: '320px',
      maxWidth: '420px',
      fontSize: '14px',
      opacity: 1,
      transform: 'translateY(0)',
      transition: 'opacity 0.6s ease-in-out 0.1s, transform 0.6s ease-in-out 0.1s',
      pointerEvents: 'none',
      fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif'
    };

    const fieldStyle = {
      marginBottom: '8px',
      display: 'flex',
      flexDirection: 'column'
    };

    const labelStyle = {
      fontWeight: '600',
      color: '#374151',
      fontSize: '12px',
      textTransform: 'uppercase',
      letterSpacing: '0.5px',
      marginBottom: '2px'
    };

    const valueStyle = {
      color: '#1f2937',
      fontSize: '14px',
      wordBreak: 'break-word'
    };

    const dividerStyle = {
      height: '1px',
      backgroundColor: '#e5e7eb',
      margin: '12px 0'
    };

    const headerStyle = {
      fontWeight: '700',
      color: '#1f2937',
      fontSize: '16px',
      marginBottom: '12px',
      borderBottom: '2px solid #e5e7eb',
      paddingBottom: '6px'
    };

    if (isLoadingDetails) {
      return (
        <div style={popupStyle}>
          <div style={headerStyle}>{document.document_type || 'Document'}</div>
          <div style={{ textAlign: 'center', color: '#6b7280', fontStyle: 'italic' }}>
            Loading document details...
          </div>
        </div>
      );
    }

    const renderElectricBillContent = (data) => (
      <>
        <div style={headerStyle}>Electric Bill</div>
        
        <div style={fieldStyle}>
          <span style={labelStyle}>Account Number</span>
          <span style={valueStyle}>{data.account_number || 'N/A'}</span>
        </div>

        <div style={fieldStyle}>
          <span style={labelStyle}>Service Address</span>
          <span style={valueStyle}>{data.service_address || data.site || 'N/A'}</span>
        </div>

        <div style={dividerStyle}></div>

        <div style={fieldStyle}>
          <span style={labelStyle}>Billing Period</span>
          <span style={valueStyle}>
            {data.billing_period_start && data.billing_period_end 
              ? `${formatDate(data.billing_period_start)} - ${formatDate(data.billing_period_end)}`
              : 'N/A'
            }
          </span>
        </div>

        <div style={fieldStyle}>
          <span style={labelStyle}>Total Usage</span>
          <span style={valueStyle}>{formatNumber(data.total_kwh, 'kWh')}</span>
        </div>

        <div style={fieldStyle}>
          <span style={labelStyle}>Total Cost</span>
          <span style={{...valueStyle, fontWeight: '600', fontSize: '16px', color: '#059669'}}>
            {formatCurrency(data.total_cost)}
          </span>
        </div>

        {data.rate_schedule && (
          <div style={fieldStyle}>
            <span style={labelStyle}>Rate Schedule</span>
            <span style={valueStyle}>{data.rate_schedule}</span>
          </div>
        )}

        {(data.demand_charge || data.energy_charge) && (
          <>
            <div style={dividerStyle}></div>
            {data.demand_charge && (
              <div style={fieldStyle}>
                <span style={labelStyle}>Demand Charge</span>
                <span style={valueStyle}>{formatCurrency(data.demand_charge)}</span>
              </div>
            )}
            {data.energy_charge && (
              <div style={fieldStyle}>
                <span style={labelStyle}>Energy Charge</span>
                <span style={valueStyle}>{formatCurrency(data.energy_charge)}</span>
              </div>
            )}
          </>
        )}
      </>
    );

    const renderWaterBillContent = (data) => (
      <>
        <div style={headerStyle}>Water Bill</div>
        
        <div style={fieldStyle}>
          <span style={labelStyle}>Account Number</span>
          <span style={valueStyle}>{data.account_number || 'N/A'}</span>
        </div>

        <div style={fieldStyle}>
          <span style={labelStyle}>Service Address</span>
          <span style={valueStyle}>{data.service_address || data.site || 'N/A'}</span>
        </div>

        <div style={dividerStyle}></div>

        <div style={fieldStyle}>
          <span style={labelStyle}>Billing Period</span>
          <span style={valueStyle}>{data.billing_period || 'N/A'}</span>
        </div>

        <div style={fieldStyle}>
          <span style={labelStyle}>Total Usage</span>
          <span style={valueStyle}>{formatNumber(data.total_gallons, 'gallons')}</span>
        </div>

        <div style={fieldStyle}>
          <span style={labelStyle}>Total Cost</span>
          <span style={{...valueStyle, fontWeight: '600', fontSize: '16px', color: '#0891b2'}}>
            {formatCurrency(data.total_cost)}
          </span>
        </div>

        {(data.base_charge || data.usage_charge || data.rate_per_gallon) && (
          <>
            <div style={dividerStyle}></div>
            {data.rate_per_gallon && (
              <div style={fieldStyle}>
                <span style={labelStyle}>Rate Per Gallon</span>
                <span style={valueStyle}>{formatCurrency(data.rate_per_gallon)}</span>
              </div>
            )}
            {data.base_charge && (
              <div style={fieldStyle}>
                <span style={labelStyle}>Base Charge</span>
                <span style={valueStyle}>{formatCurrency(data.base_charge)}</span>
              </div>
            )}
            {data.usage_charge && (
              <div style={fieldStyle}>
                <span style={labelStyle}>Usage Charge</span>
                <span style={valueStyle}>{formatCurrency(data.usage_charge)}</span>
              </div>
            )}
          </>
        )}
      </>
    );

    const renderWasteManifestContent = (data) => (
      <>
        <div style={headerStyle}>Waste Manifest</div>
        
        <div style={fieldStyle}>
          <span style={labelStyle}>Manifest Tracking Number</span>
          <span style={valueStyle}>{data.manifest_tracking_number || 'N/A'}</span>
        </div>

        {data.issue_date && (
          <div style={fieldStyle}>
            <span style={labelStyle}>Issue Date</span>
            <span style={valueStyle}>{formatDate(data.issue_date)}</span>
          </div>
        )}

        <div style={fieldStyle}>
          <span style={labelStyle}>Generator</span>
          <span style={valueStyle}>{data.generator || 'N/A'}</span>
        </div>

        <div style={fieldStyle}>
          <span style={labelStyle}>Disposal Facility</span>
          <span style={valueStyle}>{data.disposal_facility || 'N/A'}</span>
        </div>

        <div style={dividerStyle}></div>

        <div style={fieldStyle}>
          <span style={labelStyle}>Total Quantity</span>
          <span style={valueStyle}>
            {data.total_quantity !== undefined && data.weight_unit 
              ? `${formatNumber(data.total_quantity)} ${data.weight_unit}`
              : 'N/A'
            }
          </span>
        </div>

        <div style={fieldStyle}>
          <span style={labelStyle}>Total Weight</span>
          <span style={valueStyle}>
            {data.total_weight !== undefined && data.weight_unit
              ? `${formatNumber(data.total_weight)} ${data.weight_unit}`
              : 'N/A'
            }
          </span>
        </div>

        {(data.transportation_date || data.disposal_date) && (
          <>
            <div style={dividerStyle}></div>
            {data.transportation_date && (
              <div style={fieldStyle}>
                <span style={labelStyle}>Transportation Date</span>
                <span style={valueStyle}>{formatDate(data.transportation_date)}</span>
              </div>
            )}
            {data.disposal_date && (
              <div style={fieldStyle}>
                <span style={labelStyle}>Disposal Date</span>
                <span style={valueStyle}>{formatDate(data.disposal_date)}</span>
              </div>
            )}
          </>
        )}
      </>
    );

    const renderDefaultContent = (data) => (
      <>
        <div style={headerStyle}>{data.document_type || 'Document'}</div>
        
        <div style={fieldStyle}>
          <span style={labelStyle}>Document ID</span>
          <span style={valueStyle}>{data.id || 'N/A'}</span>
        </div>
        
        <div style={dividerStyle}></div>
        
        <div style={fieldStyle}>
          <span style={labelStyle}>Date Received</span>
          <span style={valueStyle}>
            {new Date(data.date_received || data.created_at).toLocaleDateString('en-US', {
              year: 'numeric',
              month: 'short',
              day: 'numeric',
              hour: '2-digit',
              minute: '2-digit'
            })}
          </span>
        </div>
        
        <div style={fieldStyle}>
          <span style={labelStyle}>Site/Location</span>
          <span style={valueStyle}>{data.site || data.location || 'Main Facility'}</span>
        </div>
      </>
    );

    const renderContent = () => {
      const data = details || document;
      const docType = (data.document_type || '').toLowerCase();

      if (docType.includes('electric')) {
        return renderElectricBillContent(data);
      } else if (docType.includes('water')) {
        return renderWaterBillContent(data);
      } else if (docType.includes('waste') || docType.includes('manifest')) {
        return renderWasteManifestContent(data);
      } else {
        return renderDefaultContent(data);
      }
    };

    return (
      <div style={popupStyle}>
        {renderContent()}
      </div>
    );
  };

  if (loading && !isIngesting) {
    return <div className="loading">Loading data...</div>;
  }

  return (
    <div>
      <h1 className="page-title">Data Management</h1>
      
      {/* Error display section commented out
      {error && (
        <div className="error">
          {error}
        </div>
      )}
      */}

      {/* Data Upload Section */}
      <h2 className="table-title">Data Upload</h2>
      <div className="card">
        <div className="table-header" style={{ display: 'flex', alignItems: 'center', gap: '20px' }}>
          <button 
            className="btn btn-primary"
            onClick={handleBulkIngestion}
            disabled={loading || isIngesting}
          >
            {isIngesting ? 'Ingesting...' : 'Ingest Invoices'}
          </button>
          {isIngesting && (
            <div style={{ flex: 1 }}>
              <IngestionProgress isIngesting={isIngesting} />
            </div>
          )}
        </div>
      </div>

      {/* Processed Documents Section */}
      <h2 className="table-title">Processed Documents ({processedDocuments.length})</h2>
      <div className="card">
        <table className="data-table">
          <thead>
            <tr>
              <th>#</th>
              <th>Date Received</th>
              <th>Site</th>
              <th>Document Type</th>
              <th>Download</th>
            </tr>
          </thead>
          <tbody>
            {processedDocuments.length === 0 ? (
              <tr>
                <td colSpan="5">No documents processed yet</td>
              </tr>
            ) : (
              processedDocuments.map((doc, index) => (
                <tr 
                  key={doc.id}
                  onMouseEnter={(e) => handleRowMouseEnter(e, index)}
                  onMouseMove={handleRowMouseMove}
                  onMouseLeave={handleRowMouseLeave}
                  style={{ 
                    cursor: 'pointer',
                    transition: 'background-color 0.2s ease',
                    backgroundColor: hoveredRow === index ? '#f8fafc' : 'transparent'
                  }}
                >
                  <td>{index + 1}</td>
                  <td>{new Date(doc.date_received || doc.created_at).toLocaleDateString('en-US', {
                    year: 'numeric',
                    month: 'short',
                    day: 'numeric',
                    hour: '2-digit',
                    minute: '2-digit'
                  })}</td>
                  <td>{doc.site || doc.location || 'Main Facility'}</td>
                  <td>{doc.document_type || doc.type || 'Unknown'}</td>
                  <td>
                    <button
                      onClick={(e) => {
                        e.stopPropagation(); // Prevent row hover popup when clicking download
                        handleDownload(doc);
                      }}
                      disabled={!doc.file_path || downloadingFiles.has(doc.id)}
                      title={doc.file_path ? "Download original document" : "Original file not available"}
                      style={{
                        background: 'none',
                        border: 'none',
                        cursor: doc.file_path && !downloadingFiles.has(doc.id) ? 'pointer' : 'not-allowed',
                        fontSize: '16px',
                        padding: '4px 8px',
                        borderRadius: '4px',
                        transition: 'background-color 0.2s ease',
                        opacity: doc.file_path ? 1 : 0.3,
                        ...(doc.file_path && !downloadingFiles.has(doc.id) && {
                          ':hover': {
                            backgroundColor: '#f3f4f6'
                          }
                        })
                      }}
                      onMouseEnter={(e) => {
                        if (doc.file_path && !downloadingFiles.has(doc.id)) {
                          e.target.style.backgroundColor = '#f3f4f6';
                        }
                      }}
                      onMouseLeave={(e) => {
                        e.target.style.backgroundColor = 'transparent';
                      }}
                    >
                      {downloadingFiles.has(doc.id) ? '⏳' : '📄'}
                    </button>
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      {/* Rejected Documents Section */}
      <h2 className="table-title">Rejected Documents ({rejectedDocuments.length})</h2>
      <div className="card">
        <table className="data-table">
          <thead>
            <tr>
              <th>#</th>
              <th>Date Rejected</th>
              <th>Original Filename</th>
              <th>Rejection Reason</th>
            </tr>
          </thead>
          <tbody>
            {rejectedDocuments.length === 0 ? (
              <tr>
                <td colSpan="4">No documents rejected yet</td>
              </tr>
            ) : (
              rejectedDocuments.map((doc, index) => (
                <tr key={index}>
                  <td>{index + 1}</td>
                  <td>{new Date(doc.date_rejected || doc.created_at).toLocaleDateString('en-US', {
                    year: 'numeric',
                    month: 'short',
                    day: 'numeric',
                    hour: '2-digit',
                    minute: '2-digit'
                  })}</td>
                  <td>{doc.file_name || doc.original_filename || doc.filename || 'Unknown'}</td>
                  <td>{doc.rejection_reason || doc.reason || 'No reason provided'}</td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
      
      {/* Hover Popup */}
      {hoveredRow !== null && processedDocuments[hoveredRow] && (
        <DocumentPopup 
          document={processedDocuments[hoveredRow]} 
          position={popupPosition}
        />
      )}
      
      <div style={{ marginTop: '20px' }}>
        <button 
          className="btn btn-secondary"
          onClick={() => {
            setProcessedDocuments([]);
            setRejectedDocuments([]);
            setDocumentDetails({});
          }}
        >
          Reset
        </button>
      </div>
    </div>
  );
};

export default DataManagement;