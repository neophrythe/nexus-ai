import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import Box from '@mui/material/Box';

import Sidebar from './components/Sidebar';
import TopBar from './components/TopBar';
import Dashboard from './pages/Dashboard';
import LiveView from './pages/LiveView';
import DebugView from './pages/DebugView';
import Plugins from './pages/Plugins';
import Training from './pages/Training';
import Vision from './pages/Vision';
import Analytics from './pages/Analytics';
import Settings from './pages/Settings';
import GameControl from './pages/GameControl';

const darkTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#00ff41',
    },
    secondary: {
      main: '#ff00ff',
    },
    background: {
      default: '#0a0a0a',
      paper: '#1a1a1a',
    },
  },
  typography: {
    fontFamily: '"Roboto Mono", "Courier New", monospace',
  },
});

function App() {
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [systemStatus, setSystemStatus] = useState(null);
  const [ws, setWs] = useState(null);

  useEffect(() => {
    // Connect to WebSocket
    const websocket = new WebSocket('ws://localhost:8000/ws');
    
    websocket.onopen = () => {
      console.log('WebSocket connected');
      setWs(websocket);
    };

    websocket.onmessage = (event) => {
      const data = JSON.parse(event.data);
      handleWebSocketMessage(data);
    };

    websocket.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    websocket.onclose = () => {
      console.log('WebSocket disconnected');
      // Reconnect after 3 seconds
      setTimeout(() => {
        window.location.reload();
      }, 3000);
    };

    // Fetch initial status
    fetchSystemStatus();

    return () => {
      if (websocket.readyState === WebSocket.OPEN) {
        websocket.close();
      }
    };
  }, []);

  const fetchSystemStatus = async () => {
    try {
      const response = await fetch('/status');
      const data = await response.json();
      setSystemStatus(data);
    } catch (error) {
      console.error('Failed to fetch status:', error);
    }
  };

  const handleWebSocketMessage = (data) => {
    switch (data.type) {
      case 'status_update':
        setSystemStatus(data.status);
        break;
      case 'stream_frame':
        // Handle in LiveView component
        break;
      default:
        console.log('Unknown message type:', data.type);
    }
  };

  const toggleSidebar = () => {
    setSidebarOpen(!sidebarOpen);
  };

  return (
    <ThemeProvider theme={darkTheme}>
      <CssBaseline />
      <Router>
        <Box sx={{ display: 'flex', height: '100vh' }}>
          <TopBar onMenuClick={toggleSidebar} systemStatus={systemStatus} />
          <Sidebar open={sidebarOpen} onClose={() => setSidebarOpen(false)} />
          
          <Box
            component="main"
            sx={{
              flexGrow: 1,
              p: 3,
              mt: 8,
              ml: sidebarOpen ? '240px' : '60px',
              transition: 'margin-left 0.3s',
              overflow: 'auto',
            }}
          >
            <Routes>
              <Route path="/" element={<Navigate to="/dashboard" />} />
              <Route path="/dashboard" element={<Dashboard ws={ws} />} />
              <Route path="/live" element={<LiveView ws={ws} />} />
              <Route path="/debug" element={<DebugView ws={ws} />} />
              <Route path="/plugins" element={<Plugins />} />
              <Route path="/training" element={<Training ws={ws} />} />
              <Route path="/vision" element={<Vision ws={ws} />} />
              <Route path="/analytics" element={<Analytics />} />
              <Route path="/game-control" element={<GameControl ws={ws} />} />
              <Route path="/settings" element={<Settings />} />
            </Routes>
          </Box>
        </Box>
      </Router>
    </ThemeProvider>
  );
}

export default App;