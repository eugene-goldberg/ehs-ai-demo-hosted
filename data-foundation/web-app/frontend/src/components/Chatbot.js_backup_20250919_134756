import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import {
  Paper,
  Box,
  TextField,
  IconButton,
  Typography,
  Avatar,
  Divider,
  CircularProgress,
  Chip
} from '@mui/material';
import {
  Send as SendIcon,
  SmartToy as SmartToyIcon,
  Person as PersonIcon,
  Clear as ClearIcon
} from '@mui/icons-material';

const Chatbot = () => {
  const [messages, setMessages] = useState([
    {
      id: 1,
      text: "Hello! I'm your EHS AI Assistant. How can I help you with environmental, health, and safety matters today?",
      sender: 'bot',
      timestamp: new Date()
    }
  ]);
  const [inputValue, setInputValue] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [sessionId, setSessionId] = useState('');
  const [apiStatus, setApiStatus] = useState('checking'); // checking, online, offline
  const messagesEndRef = useRef(null);

  // Generate session ID
  const generateSessionId = () => {
    return 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
  };

  // API health check
  const checkApiHealth = async () => {
    try {
      const response = await axios.get('http://10.136.0.4:8000/api/chatbot/health', {
        timeout: 5000
      });
      setApiStatus('online');
      console.log('API health check successful:', response.data);
    } catch (error) {
      setApiStatus('offline');
      console.error('API health check failed:', error);
    }
  };

  // Initialize component
  useEffect(() => {
    const newSessionId = generateSessionId();
    setSessionId(newSessionId);
    checkApiHealth();
    console.log('New session started:', newSessionId);
  }, []);

  // Auto-scroll to latest message
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isTyping]);

  // Handle sending messages
  const handleSendMessage = async () => {
    if (inputValue.trim() === '') return;

    const userMessage = {
      id: Date.now(),
      text: inputValue,
      sender: 'user',
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    const currentInput = inputValue;
    setInputValue('');
    setIsTyping(true);

    try {
      // Make API call to backend
      const response = await axios.post('http://10.136.0.4:8000/api/chatbot/chat', {
        message: currentInput,
        session_id: sessionId
      }, {
        timeout: 30000, // 30 second timeout
        headers: {
          'Content-Type': 'application/json'
        }
      });
      
      // Add bot response
      const botMessage = {
        id: Date.now() + 1,
        text: response.data.response || 'I received your message but encountered an issue generating a response.',
        sender: 'bot',
        timestamp: new Date()
      };
      setMessages(prev => [...prev, botMessage]);
      
    } catch (error) {
      console.error('API call failed:', error);
      
      // Handle different types of errors
      let errorText = 'Sorry, I encountered an error. Please try again.';
      
      if (error.code === 'ECONNABORTED') {
        errorText = 'Request timed out. Please try again with a shorter message.';
      } else if (error.response) {
        // Server responded with error status
        errorText = `Server error (${error.response.status}): ${error.response.data?.detail || 'Please try again later.'}`;
      } else if (error.request) {
        // Network error
        errorText = 'Unable to connect to the AI service. Please check your connection and try again.';
        setApiStatus('offline');
      }
      
      const errorMessage = {
        id: Date.now() + 1,
        text: errorText,
        sender: 'bot',
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsTyping(false);
    }
  };

  // Clear session and start fresh
  const handleClearSession = () => {
    const newSessionId = generateSessionId();
    setSessionId(newSessionId);
    setMessages([
      {
        id: Date.now(),
        text: "Hello! I'm your EHS AI Assistant. How can I help you with environmental, health, and safety matters today?",
        sender: 'bot',
        timestamp: new Date()
      }
    ]);
    console.log('Session cleared, new session started:', newSessionId);
  };

  // Handle Enter key press
  const handleKeyPress = (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      handleSendMessage();
    }
  };

  // Format timestamp
  const formatTime = (timestamp) => {
    return timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  // Get status color
  const getStatusColor = () => {
    switch (apiStatus) {
      case 'online': return 'success';
      case 'offline': return 'error';
      case 'checking': return 'warning';
      default: return 'default';
    }
  };

  // Get status text
  const getStatusText = () => {
    switch (apiStatus) {
      case 'online': return 'AI Online';
      case 'offline': return 'AI Offline';
      case 'checking': return 'Connecting...';
      default: return 'Unknown';
    }
  };

  return (
    <Paper
      elevation={3}
      sx={{
        height: 450,
        display: 'flex',
        flexDirection: 'column',
        borderRadius: 2,
        overflow: 'hidden',
        bgcolor: 'background.paper'
      }}
    >
      {/* Header */}
      <Box
        sx={{
          p: 2,
          bgcolor: 'primary.main',
          color: 'primary.contrastText',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between'
        }}
      >
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <SmartToyIcon />
          <Typography variant="h6" component="h2">
            EHS AI Assistant
          </Typography>
        </Box>
        
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Chip
            label={getStatusText()}
            color={getStatusColor()}
            size="small"
            variant="filled"
            sx={{ 
              color: 'white',
              '& .MuiChip-label': { fontSize: '0.75rem' }
            }}
          />
          <IconButton
            onClick={handleClearSession}
            sx={{ color: 'white' }}
            title="Clear conversation"
            size="small"
          >
            <ClearIcon />
          </IconButton>
        </Box>
      </Box>

      <Divider />

      {/* Messages Area */}
      <Box
        sx={{
          flex: 1,
          p: 1,
          overflowY: 'auto',
          display: 'flex',
          flexDirection: 'column',
          gap: 1,
          bgcolor: '#fafafa'
        }}
      >
        {messages.map((message) => (
          <Box
            key={message.id}
            sx={{
              display: 'flex',
              flexDirection: message.sender === 'user' ? 'row-reverse' : 'row',
              alignItems: 'flex-start',
              gap: 1,
              mb: 1
            }}
          >
            <Avatar
              sx={{
                width: 32,
                height: 32,
                bgcolor: message.sender === 'user' ? 'primary.main' : 'secondary.main'
              }}
            >
              {message.sender === 'user' ? <PersonIcon /> : <SmartToyIcon />}
            </Avatar>
            
            <Box
              sx={{
                maxWidth: '70%',
                minWidth: '100px'
              }}
            >
              <Paper
                elevation={1}
                sx={{
                  p: 1.5,
                  bgcolor: message.sender === 'user' ? 'primary.light' : 'white',
                  color: message.sender === 'user' ? 'primary.contrastText' : 'text.primary',
                  borderRadius: 2,
                  borderTopLeftRadius: message.sender === 'user' ? 2 : 0.5,
                  borderTopRightRadius: message.sender === 'user' ? 0.5 : 2
                }}
              >
                <Typography variant="body2" sx={{ wordBreak: 'break-word' }}>
                  {message.text}
                </Typography>
                <Typography
                  variant="caption"
                  sx={{
                    display: 'block',
                    mt: 0.5,
                    opacity: 0.7,
                    fontSize: '0.7rem'
                  }}
                >
                  {formatTime(message.timestamp)}
                </Typography>
              </Paper>
            </Box>
          </Box>
        ))}

        {/* Typing Indicator */}
        {isTyping && (
          <Box
            sx={{
              display: 'flex',
              alignItems: 'flex-start',
              gap: 1,
              mb: 1
            }}
          >
            <Avatar
              sx={{
                width: 32,
                height: 32,
                bgcolor: 'secondary.main'
              }}
            >
              <SmartToyIcon />
            </Avatar>
            
            <Paper
              elevation={1}
              sx={{
                p: 1.5,
                bgcolor: 'white',
                borderRadius: 2,
                borderTopLeftRadius: 0.5,
                display: 'flex',
                alignItems: 'center',
                gap: 1
              }}
            >
              <CircularProgress size={16} />
              <Typography variant="body2" sx={{ fontStyle: 'italic' }}>
                AI is thinking...
              </Typography>
            </Paper>
          </Box>
        )}

        <div ref={messagesEndRef} />
      </Box>

      <Divider />

      {/* Input Area */}
      <Box
        sx={{
          p: 2,
          display: 'flex',
          gap: 1,
          bgcolor: 'background.paper'
        }}
      >
        <TextField
          fullWidth
          variant="outlined"
          placeholder={apiStatus === 'offline' ? 'AI service is offline...' : 'Type your EHS question here...'}
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onKeyPress={handleKeyPress}
          size="small"
          multiline
          maxRows={3}
          disabled={isTyping || apiStatus === 'offline'}
          sx={{
            '& .MuiOutlinedInput-root': {
              borderRadius: 2
            }
          }}
        />
        <IconButton
          onClick={handleSendMessage}
          disabled={!inputValue.trim() || isTyping || apiStatus === 'offline'}
          color="primary"
          sx={{
            bgcolor: 'primary.main',
            color: 'white',
            '&:hover': {
              bgcolor: 'primary.dark'
            },
            '&:disabled': {
              bgcolor: 'grey.300'
            }
          }}
        >
          <SendIcon />
        </IconButton>
      </Box>
    </Paper>
  );
};

export default Chatbot;
