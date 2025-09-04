import React, { useState, useEffect } from 'react';
import axios from 'axios';
import IngestionProgress from './IngestionProgress';
import { API_ENDPOINTS } from '../config/api';
import apiConfig from '../config/api';

const DataManagement = () => {
  const [processedDocuments, setProcessedDocuments] = useState([]);
  const [rejectedDocuments, setRejectedDocuments] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [isIngesting, setIsIngesting] = useState(false);
  const [downloadingFiles, setDownloadingFiles] = useState(new Set());
  const [isAiEngineRoomExpanded, setIsAiEngineRoomExpanded] = useState(false);
  
  // Transcript state - updated to handle markdown content
  const [transcriptMarkdown, setTranscriptMarkdown] = useState('');
  const [transcriptLoading, setTranscriptLoading] = useState(false);
  const [transcriptError, setTranscriptError] = useState(null);
  const [hasTranscriptData, setHasTranscriptData] = useState(false);
  
  // Popup state
  const [hoveredRow, setHoveredRow] = useState(null);
  const [popupPosition, setPopupPosition] = useState({ x: 0, y: 0 });
  const [documentDetails, setDocumentDetails] = useState({});
  const [loadingDetails, setLoadingDetails] = useState({});

  useEffect(() => {
    fetchData();
  }, []);

  // Fetch transcript data when AI Engine Room is expanded
  useEffect(() => {
    if (isAiEngineRoomExpanded && !hasTranscriptData) {
      fetchTranscriptData();
    }
  }, [isAiEngineRoomExpanded, hasTranscriptData]);

  const fetchData = async () => {
    try {
      setLoading(true);
      const [documentsRes, rejectedRes] = await Promise.all([
        axios.get(API_ENDPOINTS.processedDocuments),
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
      
      // Fetch transcript data after documents are loaded if AI Engine Room is expanded
      if (isAiEngineRoomExpanded && !hasTranscriptData) {
        await fetchTranscriptData();
      }
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

  const fetchTranscriptData = async () => {
    // Don't fetch if we already have transcript data
    if (hasTranscriptData) {
      return;
    }

    try {
      setTranscriptLoading(true);
      setTranscriptError(null);
      
      console.log('Fetching transcript markdown from new backend endpoint...');
      
      // Call the conversations endpoint which now returns markdown content
      const conversationsResponse = await axios.get(apiConfig.langsmith.conversations);
      
      // Check if the response has the expected markdown_content format
      if (!conversationsResponse.data) {
        setTranscriptError('No response data received from server');
        setTranscriptMarkdown('');
        return;
      }
      
      // Handle the new format: { "markdown_content": "..." }
      if (conversationsResponse.data.markdown_content) {
        const markdownContent = conversationsResponse.data.markdown_content;
        console.log(`Received markdown content (${markdownContent.length} characters)`);
        
        // Set the markdown content directly
        setTranscriptMarkdown(markdownContent);
        setHasTranscriptData(true);
        
      } else if (conversationsResponse.data.error) {
        // Handle error response from backend
        setTranscriptError(`Backend error: ${conversationsResponse.data.error}`);
        setTranscriptMarkdown('');
        
      } else {
        // Handle unexpected response format
        setTranscriptError('Unexpected response format from server. Expected markdown_content field.');
        setTranscriptMarkdown('');
        console.warn('Unexpected response format:', conversationsResponse.data);
      }
      
    } catch (err) {
      console.error('Error fetching transcript data:', err);
      let errorMessage = 'Failed to load transcript data from conversations API. ';
      
      if (err.response?.status === 404) {
        errorMessage += 'Conversations endpoint not found. Please ensure the backend is running.';
      } else if (err.response?.status >= 500) {
        errorMessage += 'Server error occurred. Please try again later.';
      } else if (err.response?.data?.error) {
        errorMessage += `Server responded: ${err.response.data.error}`;
      } else if (err.code === 'ECONNREFUSED') {
        errorMessage += 'Cannot connect to backend server.';
      } else {
        errorMessage += 'Please try again.';
      }
      
      setTranscriptError(errorMessage);
      setTranscriptMarkdown('');
    } finally {
      setTranscriptLoading(false);
    }
  };

  const fetchDocumentDetails = async (documentId) => {
    if (documentDetails[documentId] || loadingDetails[documentId]) {
      return; // Already loaded or loading
    }

    try {
      setLoadingDetails(prev => ({ ...prev, [documentId]: true }));
      const response = await axios.get(API_ENDPOINTS.processedDocumentById(documentId));
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
      
      const response = await axios.get(`${API_ENDPOINTS.processedDocuments.replace('/processed-documents', '')}/documents/${doc.id}/download`, {
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
    // Clear transcript flag when starting new ingestion
    setHasTranscriptData(false);
    
    try {
      const response = await axios.post(API_ENDPOINTS.ingestion, {
        clear_database: true
      });
      
      if (response.data.status === 'completed') {
        // Fixed line: Added null checking for response.data.data
        const ingestionData = response.data.data || {};
        const successfulIngestions = ingestionData.successful_ingestions || 0;
        console.log(`Ingestion completed successfully! ${successfulIngestions}/3 documents processed.`);
        
        // Handle LangSmith traces if available
        if (ingestionData.langsmith_traces && ingestionData.langsmith_traces.traces) {
          const traces = ingestionData.langsmith_traces.traces;
          console.log(`Found ${traces.length} LangSmith traces`);
          
          // Transform LangSmith traces into transcript format
          const formattedTranscript = traces
            .filter(trace => trace.run_type === 'llm' && trace.inputs && trace.outputs)
            .map((trace, index) => {
              // Extract user message from inputs
              let userMessage = '';
              let assistantMessage = '';
              
              try {
                // Handle LangSmith trace input format: inputs.messages array
                if (trace.inputs.messages && Array.isArray(trace.inputs.messages)) {
                  // Extract the last user message from the messages array
                  const userMessages = trace.inputs.messages.filter(msg => 
                    msg.role === 'user' || msg.type === 'user' || 
                    (msg.content && typeof msg.content === 'string')
                  );
                  
                  if (userMessages.length > 0) {
                    const lastUserMessage = userMessages[userMessages.length - 1];
                    // Handle different message content structures
                    if (typeof lastUserMessage.content === 'string') {
                      userMessage = lastUserMessage.content;
                    } else if (lastUserMessage.content && Array.isArray(lastUserMessage.content)) {
                      // Handle content arrays (for multimodal messages)
                      const textContent = lastUserMessage.content
                        .filter(item => item.type === 'text')
                        .map(item => item.text)
                        .join(' ');
                      userMessage = textContent;
                    } else if (typeof lastUserMessage === 'string') {
                      userMessage = lastUserMessage;
                    }
                  }
                } else if (trace.inputs.input) {
                  // Fallback to direct input field
                  userMessage = trace.inputs.input;
                } else if (trace.inputs.prompt) {
                  // Another fallback for prompt field
                  userMessage = trace.inputs.prompt;
                }
                
                // Handle LangSmith trace output format: outputs.generations array
                if (trace.outputs && trace.outputs.generations && Array.isArray(trace.outputs.generations)) {
                  // Extract the assistant response from generations
                  if (trace.outputs.generations.length > 0) {
                    const generation = trace.outputs.generations[0];
                    if (Array.isArray(generation) && generation.length > 0) {
                      // Handle nested array structure: generations[0][0].text
                      const firstGeneration = generation[0];
                      if (firstGeneration && firstGeneration.text) {
                        assistantMessage = firstGeneration.text;
                      } else if (firstGeneration && firstGeneration.message && firstGeneration.message.content) {
                        assistantMessage = firstGeneration.message.content;
                      } else if (typeof firstGeneration === 'string') {
                        assistantMessage = firstGeneration;
                      }
                    } else if (generation.text) {
                      // Handle direct generation object with text
                      assistantMessage = generation.text;
                    } else if (generation.message && generation.message.content) {
                      // Handle generation with message object
                      assistantMessage = generation.message.content;
                    } else if (typeof generation === 'string') {
                      assistantMessage = generation;
                    }
                  }
                } else if (trace.outputs && trace.outputs.output) {
                  // Fallback to direct output field
                  assistantMessage = trace.outputs.output;
                } else if (trace.outputs && trace.outputs.content) {
                  // Another fallback for content field
                  assistantMessage = trace.outputs.content;
                }
                
                // Removed debug console.log statements here
                
              } catch (e) {
                console.error('Error parsing trace:', e, trace);
              }
              
              // Return both user and assistant messages
              const messages = [];
              if (userMessage && userMessage.trim()) {
                messages.push({
                  id: `${trace.id}_user`,
                  role: 'user',
                  content: userMessage.trim(),
                  timestamp: trace.start_time,
                  context: {
                    trace_id: trace.id,
                    model: trace.extra?.invocation_params?.model || trace.extra?.metadata?.model || 'unknown',
                    operation: trace.name || 'LLM Call'
                  }
                });
              }
              if (assistantMessage && assistantMessage.trim()) {
                messages.push({
                  id: `${trace.id}_assistant`,
                  role: 'assistant',
                  content: assistantMessage.trim(),
                  timestamp: trace.end_time,
                  context: {
                    trace_id: trace.id,
                    model: trace.extra?.invocation_params?.model || trace.extra?.metadata?.model || 'unknown',
                    latency: trace.end_time && trace.start_time ? 
                      ((new Date(trace.end_time) - new Date(trace.start_time)) / 1000).toFixed(2) + 's' : 
                      'unknown'
                  }
                });
              }
              return messages;
            })
            .flat()
            .filter(msg => msg.content && msg.content.trim()); // Filter out empty messages
          
          console.log(`Formatted ${formattedTranscript.length} transcript messages from ${traces.length} traces`);
          
          // Fetch ingestion instructions and prepend to transcript
          try {
            const instructionsResponse = await axios.get('http://localhost:8001/api/ingestion-instructions');
            if (instructionsResponse.data && instructionsResponse.data.content) {
              // Create system message with instructions
              const systemMessage = {
                id: 'system_instructions',
                role: 'system',
                content: instructionsResponse.data.content,
                timestamp: new Date().toISOString(),
                context: {
                  operation: 'Ingestion Instructions',
                  source: 'System'
                }
              };
              
              // Prepend system message to transcript
              formattedTranscript.unshift(systemMessage);
              console.log('Prepended ingestion instructions to transcript');
            }
          } catch (instructionsError) {
            console.warn('Failed to fetch ingestion instructions:', instructionsError);
            // Continue without instructions - graceful error handling
          }
          
          // Convert formatted transcript to markdown-like format for legacy compatibility
          const markdownLikeContent = formattedTranscript.map((msg, idx) => {
            const timestamp = msg.timestamp ? new Date(msg.timestamp).toLocaleString() : 'Unknown time';
            const model = msg.context?.model || 'unknown';
            const role = msg.role?.charAt(0).toUpperCase() + msg.role?.slice(1) || 'Unknown';
            
            return `## ${role} Message ${idx + 1}\n**Time:** ${timestamp}\n**Model:** ${model}\n\n${msg.content}\n\n---\n`;
          }).join('\n');
          
          // Update transcript data and set flag
          setTranscriptMarkdown(markdownLikeContent);
          setHasTranscriptData(true);
          
          // Commented out: Automatic expansion of AI Engine Room when transcript data arrives
          // if (formattedTranscript.length > 0) {
          //   setIsAiEngineRoomExpanded(true);
          // }
        }
        
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

  const toggleAiEngineRoom = () => {
    setIsAiEngineRoomExpanded(!isAiEngineRoomExpanded);
  };

  const handleRefreshTranscript = async () => {
    // Clear the transcript flag to allow fetching fresh data
    setHasTranscriptData(false);
    setTranscriptMarkdown('');
    await fetchTranscriptData();
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

  const formatTimestamp = (timestamp) => {
    if (!timestamp) return '';
    const date = new Date(timestamp);
    return date.toLocaleTimeString('en-US', {
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
      hour12: false
    });
  };

  const getMessageRoleStyle = (role) => {
    const baseStyle = {
      display: 'inline-block',
      padding: '2px 8px',
      borderRadius: '12px',
      fontSize: '12px',
      fontWeight: '600',
      textTransform: 'uppercase',
      letterSpacing: '0.5px',
      marginRight: '8px'
    };

    switch (role?.toLowerCase()) {
      case 'system':
        return { ...baseStyle, backgroundColor: '#fef3c7', color: '#92400e' };
      case 'assistant':
        return { ...baseStyle, backgroundColor: '#dbeafe', color: '#1e40af' };
      case 'user':
        return { ...baseStyle, backgroundColor: '#dcfce7', color: '#166534' };
      default:
        return { ...baseStyle, backgroundColor: '#f3f4f6', color: '#374151' };
    }
  };

  const getMessageBubbleStyle = (role) => {
    const baseStyle = {
      padding: '12px 16px',
      marginBottom: '8px',
      borderRadius: '12px',
      maxWidth: '80%',
      wordBreak: 'break-word',
      fontSize: '14px',
      lineHeight: '1.5'
    };

    switch (role?.toLowerCase()) {
      case 'system':
        return { 
          ...baseStyle, 
          backgroundColor: '#fffbeb', 
          border: '1px solid #fef3c7',
          alignSelf: 'flex-start'
        };
      case 'assistant':
        return { 
          ...baseStyle, 
          backgroundColor: '#eff6ff', 
          border: '1px solid #dbeafe',
          alignSelf: 'flex-start'
        };
      case 'user':
        return { 
          ...baseStyle, 
          backgroundColor: '#f0fdf4', 
          border: '1px solid #dcfce7',
          alignSelf: 'flex-end'
        };
      default:
        return { 
          ...baseStyle, 
          backgroundColor: '#f9fafb', 
          border: '1px solid #f3f4f6',
          alignSelf: 'flex-start'
        };
    }
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

  // Simple function to render markdown-like content as basic HTML
  const renderMarkdownContent = (markdown) => {
    if (!markdown || typeof markdown !== 'string') {
      return <div>No content available</div>;
    }

    // Split content into lines and process basic markdown-like formatting
    const lines = markdown.split('\n');
    const elements = [];
    let currentIdx = 0;

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      
      // Handle headers
      if (line.startsWith('## ')) {
        elements.push(
          <h3 key={currentIdx++} style={{ 
            fontSize: '16px', 
            fontWeight: 'bold', 
            marginTop: '20px', 
            marginBottom: '10px',
            color: '#1f2937'
          }}>
            {line.substring(3)}
          </h3>
        );
      }
      // Handle bold text (simple **text** format)
      else if (line.startsWith('**') && line.endsWith('**')) {
        elements.push(
          <div key={currentIdx++} style={{ 
            fontWeight: 'bold', 
            marginBottom: '5px',
            color: '#374151'
          }}>
            {line.substring(2, line.length - 2)}
          </div>
        );
      }
      // Handle horizontal rules
      else if (line.trim() === '---') {
        elements.push(
          <hr key={currentIdx++} style={{ 
            border: 'none', 
            borderTop: '1px solid #e5e7eb', 
            margin: '15px 0' 
          }} />
        );
      }
      // Handle regular content
      else if (line.trim() !== '') {
        elements.push(
          <div key={currentIdx++} style={{ 
            marginBottom: '8px',
            lineHeight: '1.6',
            color: '#1f2937',
            whiteSpace: 'pre-wrap'
          }}>
            {line}
          </div>
        );
      }
      // Handle empty lines (add spacing)
      else {
        elements.push(<div key={currentIdx++} style={{ height: '10px' }} />);
      }
    }

    return <div>{elements}</div>;
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
          <span style={labelStyle}>Prorated Monthly Usage</span>
          <span style={valueStyle}>{formatNumber(data.prorated_monthly_usage || data.total_kwh, 'kWh')}</span>
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

      {/* LLM Transcript Section */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginTop: '30px' }}>
        <h2 className="table-title">LLM Interaction Transcript</h2>
        <button
          onClick={toggleAiEngineRoom}
          style={{
            background: 'none',
            border: 'none',
            cursor: 'pointer',
            fontSize: '18px',
            padding: '4px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            borderRadius: '4px',
            transition: 'background-color 0.2s ease',
            color: '#6b7280'
          }}
          title={isAiEngineRoomExpanded ? 'Collapse Transcript' : 'Expand Transcript'}
          onMouseEnter={(e) => {
            e.target.style.backgroundColor = '#f3f4f6';
          }}
          onMouseLeave={(e) => {
            e.target.style.backgroundColor = 'transparent';
          }}
        >
          {isAiEngineRoomExpanded ? '▼' : '▶'}
        </button>
        {hasTranscriptData && (
          <div style={{
            fontSize: '12px',
            color: '#059669',
            backgroundColor: '#d1fae5',
            padding: '2px 8px',
            borderRadius: '4px',
            fontWeight: '500'
          }}>
            Transcript Data
          </div>
        )}
      </div>
      
      {isAiEngineRoomExpanded && (
        <div className="card" style={{ marginTop: '10px' }}>
          <div style={{ padding: '20px' }}>
            {/* Refresh Button */}
            <div style={{ 
              display: 'flex', 
              justifyContent: 'flex-end',
              marginBottom: '16px'
            }}>
              <button
                onClick={handleRefreshTranscript}
                disabled={transcriptLoading}
                style={{
                  backgroundColor: '#f3f4f6',
                  color: '#374151',
                  border: '1px solid #d1d5db',
                  borderRadius: '6px',
                  padding: '8px 16px',
                  fontSize: '14px',
                  fontWeight: '500',
                  cursor: transcriptLoading ? 'not-allowed' : 'pointer',
                  transition: 'all 0.2s ease',
                  opacity: transcriptLoading ? 0.6 : 1
                }}
                onMouseEnter={(e) => {
                  if (!transcriptLoading) {
                    e.target.style.backgroundColor = '#e5e7eb';
                  }
                }}
                onMouseLeave={(e) => {
                  if (!transcriptLoading) {
                    e.target.style.backgroundColor = '#f3f4f6';
                  }
                }}
              >
                {transcriptLoading ? '🔄 Refreshing...' : '🔄 Refresh'}
              </button>
            </div>

            {/* Transcript Display */}
            <div style={{
              border: '1px solid #e5e7eb',
              borderRadius: '8px',
              backgroundColor: '#fafafa',
              minHeight: '300px',
              maxHeight: '500px',
              overflow: 'hidden',
              display: 'flex',
              flexDirection: 'column'
            }}>
              {transcriptLoading ? (
                <div style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  height: '300px',
                  color: '#6b7280',
                  fontSize: '16px'
                }}>
                  Loading transcript...
                </div>
              ) : transcriptError ? (
                <div style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  height: '300px',
                  color: '#dc2626',
                  fontSize: '16px',
                  textAlign: 'center',
                  padding: '20px'
                }}>
                  {transcriptError}
                </div>
              ) : !transcriptMarkdown ? (
                <div style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  height: '300px',
                  color: '#6b7280',
                  fontSize: '16px',
                  textAlign: 'center'
                }}>
                  No transcript data available yet.<br />
                  Run the ingestion process to see LLM interactions.
                </div>
              ) : (
                <div style={{
                  flex: 1,
                  overflowY: 'auto',
                  padding: '16px'
                }}>
                  {renderMarkdownContent(transcriptMarkdown)}
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Processed Documents Section */}
      <h2 className="table-title" style={{ marginTop: '40px' }}>Processed Documents ({processedDocuments.length})</h2>
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
            setHasTranscriptData(false);
            setTranscriptMarkdown('');
          }}
        >
          Reset
        </button>
      </div>
    </div>
  );
};

export default DataManagement;