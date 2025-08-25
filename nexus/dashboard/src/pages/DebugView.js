import React, { useState, useEffect, useRef } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Grid,
  Button,
  TextField,
  Switch,
  FormControlLabel,
  List,
  ListItem,
  ListItemText,
  Chip,
  Slider,
  Alert,
  Tab,
  Tabs,
  Paper,
  IconButton,
  Tooltip
} from '@mui/material';
import {
  PlayArrow,
  Stop,
  Camera,
  Settings,
  Refresh,
  Download,
  History,
  BugReport,
  Visibility,
  VisibilityOff
} from '@mui/icons-material';

function TabPanel({ children, value, index, ...other }) {
  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`debug-tabpanel-${index}`}
      aria-labelledby={`debug-tab-${index}`}
      {...other}
    >
      {value === index && (
        <Box sx={{ p: 3 }}>
          {children}
        </Box>
      )}
    </div>
  );
}

export default function DebugView() {
  // State management
  const [connected, setConnected] = useState(false);
  const [websocket, setWebsocket] = useState(null);
  const [sessionId, setSessionId] = useState('default');
  const [currentFrame, setCurrentFrame] = useState(null);
  const [frameHistory, setFrameHistory] = useState([]);
  const [sessions, setSessions] = useState({});
  const [streaming, setStreaming] = useState(false);
  const [tabValue, setTabValue] = useState(0);
  
  // Debug settings
  const [debugSettings, setDebugSettings] = useState({
    show_detections: true,
    show_ocr: true,
    show_sprites: true,
    detection_confidence_threshold: 0.5,
    annotation_color: [0, 255, 0],
    font_scale: 0.7
  });
  
  // Statistics
  const [stats, setStats] = useState({
    frames_processed: 0,
    total_detections: 0,
    total_ocr_results: 0,
    avg_processing_time_ms: 0,
    session_duration_s: 0
  });
  
  // UI State
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const canvasRef = useRef(null);
  
  // WebSocket connection
  useEffect(() => {
    const ws = new WebSocket('ws://localhost:8000/ws');
    
    ws.onopen = () => {
      setConnected(true);
      setWebsocket(ws);
      console.log('WebSocket connected for debugging');
      
      // Initialize debug session
      createDebugSession();
    };
    
    ws.onclose = () => {
      setConnected(false);
      setWebsocket(null);
      console.log('WebSocket disconnected');
    };
    
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      handleWebSocketMessage(data);
    };
    
    ws.onerror = (error) => {
      setError('WebSocket connection failed');
      console.error('WebSocket error:', error);
    };
    
    return () => {
      ws.close();
    };
  }, []);
  
  // Handle WebSocket messages
  const handleWebSocketMessage = (data) => {
    switch (data.type) {
      case 'debug_frame':
        setCurrentFrame(data.frame_data);
        setFrameHistory(prev => [...prev.slice(-19), data.frame_data]); // Keep last 20 frames
        updateCanvas(data.frame_data.image);
        
        if (data.frame_data.stats) {
          setStats(data.frame_data.stats);
        }
        break;
      
      case 'debug_settings_updated':
        if (data.success) {
          console.log('Debug settings updated successfully');
        } else {
          setError('Failed to update debug settings');
        }
        break;
      
      case 'error':
        setError(data.message);
        break;
        
      default:
        console.log('Unknown message type:', data.type);
    }
  };
  
  // API calls
  const createDebugSession = async () => {
    try {
      const response = await fetch('http://localhost:8000/debug/session', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId, max_frames: 50 })
      });
      const data = await response.json();
      setSessionId(data.session_id);
      console.log('Debug session created:', data.session_id);
    } catch (err) {
      setError('Failed to create debug session');
    }
  };
  
  const loadSessions = async () => {
    try {
      const response = await fetch('http://localhost:8000/debug/sessions');
      const data = await response.json();
      setSessions(data.sessions || {});
    } catch (err) {
      setError('Failed to load sessions');
    }
  };
  
  const updateSettings = async (newSettings) => {
    if (!websocket) return;
    
    websocket.send(JSON.stringify({
      type: 'debug_settings',
      session_id: sessionId,
      settings: newSettings
    }));
    
    setDebugSettings(prev => ({ ...prev, ...newSettings }));
  };
  
  // Canvas rendering
  const updateCanvas = (imageBase64) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    const img = new Image();
    
    img.onload = () => {
      // Calculate scaling to fit canvas
      const maxWidth = canvas.width;
      const maxHeight = canvas.height;
      const scale = Math.min(maxWidth / img.width, maxHeight / img.height);
      
      const scaledWidth = img.width * scale;
      const scaledHeight = img.height * scale;
      const x = (maxWidth - scaledWidth) / 2;
      const y = (maxHeight - scaledHeight) / 2;
      
      // Clear canvas and draw image
      ctx.clearRect(0, 0, maxWidth, maxHeight);
      ctx.drawImage(img, x, y, scaledWidth, scaledHeight);
    };
    
    img.src = `data:image/jpeg;base64,${imageBase64}`;
  };
  
  // Control functions
  const captureFrame = () => {
    if (!websocket) return;
    
    websocket.send(JSON.stringify({
      type: 'debug_capture',
      session_id: sessionId
    }));
  };
  
  const toggleStreaming = () => {
    if (streaming) {
      setStreaming(false);
    } else {
      setStreaming(true);
      streamFrames();
    }
  };
  
  const streamFrames = () => {
    if (!streaming || !websocket) return;
    
    captureFrame();
    setTimeout(() => {
      if (streaming) streamFrames();
    }, 1000 / 10); // 10 FPS
  };
  
  const downloadFrame = () => {
    if (!currentFrame) return;
    
    const link = document.createElement('a');
    link.href = `data:image/jpeg;base64,${currentFrame.image}`;
    link.download = `debug_frame_${currentFrame.frame_id}.jpg`;
    link.click();
  };
  
  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        Visual Debugger
      </Typography>
      
      {/* Connection Status */}
      <Alert 
        severity={connected ? "success" : "error"} 
        sx={{ mb: 2 }}
      >
        {connected ? `Connected - Session: ${sessionId}` : "Disconnected"}
      </Alert>
      
      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}
      
      {/* Controls */}
      <Paper sx={{ p: 2, mb: 2 }}>
        <Box sx={{ display: 'flex', gap: 2, alignItems: 'center', flexWrap: 'wrap' }}>
          <Button
            variant="contained"
            startIcon={<Camera />}
            onClick={captureFrame}
            disabled={!connected}
          >
            Capture
          </Button>
          
          <Button
            variant={streaming ? "outlined" : "contained"}
            startIcon={streaming ? <Stop /> : <PlayArrow />}
            onClick={toggleStreaming}
            disabled={!connected}
            color={streaming ? "secondary" : "primary"}
          >
            {streaming ? "Stop Stream" : "Start Stream"}
          </Button>
          
          <Button
            startIcon={<Download />}
            onClick={downloadFrame}
            disabled={!currentFrame}
          >
            Save Frame
          </Button>
          
          <Button
            startIcon={<Refresh />}
            onClick={loadSessions}
          >
            Refresh
          </Button>
          
          <TextField
            label="Session ID"
            value={sessionId}
            onChange={(e) => setSessionId(e.target.value)}
            size="small"
            sx={{ width: 150 }}
          />
        </Box>
      </Paper>
      
      <Grid container spacing={3}>
        {/* Main Display */}
        <Grid item xs={12} md={8}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Live Debug View
              </Typography>
              <Box 
                sx={{ 
                  display: 'flex', 
                  justifyContent: 'center',
                  backgroundColor: '#000',
                  p: 1,
                  borderRadius: 1
                }}
              >
                <canvas
                  ref={canvasRef}
                  width={800}
                  height={600}
                  style={{ maxWidth: '100%', height: 'auto' }}
                />
              </Box>
              
              {currentFrame && (
                <Box sx={{ mt: 2, display: 'flex', gap: 2, flexWrap: 'wrap' }}>
                  <Chip 
                    label={`Frame ${currentFrame.frame_id}`} 
                    color="primary" 
                  />
                  <Chip 
                    label={`${currentFrame.shape[1]}x${currentFrame.shape[0]}`} 
                  />
                  <Chip 
                    label={`${currentFrame.processing_time_ms.toFixed(2)}ms`} 
                    color="secondary"
                  />
                  <Chip 
                    label={`${currentFrame.detections.length} objects`} 
                  />
                  <Chip 
                    label={`${currentFrame.ocr_results.length} text`} 
                  />
                </Box>
              )}
            </CardContent>
          </Card>
        </Grid>
        
        {/* Side Panel */}
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Tabs value={tabValue} onChange={(e, v) => setTabValue(v)}>
                <Tab label="Settings" />
                <Tab label="Objects" />
                <Tab label="Text" />
                <Tab label="Stats" />
              </Tabs>
              
              {/* Settings Tab */}
              <TabPanel value={tabValue} index={0}>
                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={debugSettings.show_detections}
                        onChange={(e) => updateSettings({ show_detections: e.target.checked })}
                      />
                    }
                    label="Show Object Detection"
                  />
                  
                  <FormControlLabel
                    control={
                      <Switch
                        checked={debugSettings.show_ocr}
                        onChange={(e) => updateSettings({ show_ocr: e.target.checked })}
                      />
                    }
                    label="Show OCR Results"
                  />
                  
                  <FormControlLabel
                    control={
                      <Switch
                        checked={debugSettings.show_sprites}
                        onChange={(e) => updateSettings({ show_sprites: e.target.checked })}
                      />
                    }
                    label="Show Sprites"
                  />
                  
                  <Box>
                    <Typography gutterBottom>
                      Confidence Threshold: {debugSettings.detection_confidence_threshold}
                    </Typography>
                    <Slider
                      value={debugSettings.detection_confidence_threshold}
                      onChange={(e, v) => updateSettings({ detection_confidence_threshold: v })}
                      min={0}
                      max={1}
                      step={0.1}
                      valueLabelDisplay="auto"
                    />
                  </Box>
                  
                  <Box>
                    <Typography gutterBottom>
                      Font Scale: {debugSettings.font_scale}
                    </Typography>
                    <Slider
                      value={debugSettings.font_scale}
                      onChange={(e, v) => updateSettings({ font_scale: v })}
                      min={0.3}
                      max={2.0}
                      step={0.1}
                      valueLabelDisplay="auto"
                    />
                  </Box>
                </Box>
              </TabPanel>
              
              {/* Objects Tab */}
              <TabPanel value={tabValue} index={1}>
                <List>
                  {currentFrame?.detections?.map((detection, index) => (
                    <ListItem key={index}>
                      <ListItemText
                        primary={detection.class || 'Unknown'}
                        secondary={`Confidence: ${(detection.confidence * 100).toFixed(1)}%`}
                      />
                    </ListItem>
                  )) || <Typography color="textSecondary">No detections</Typography>}
                </List>
              </TabPanel>
              
              {/* Text Tab */}
              <TabPanel value={tabValue} index={2}>
                <List>
                  {currentFrame?.ocr_results?.map((result, index) => (
                    <ListItem key={index}>
                      <ListItemText
                        primary={result.text || 'No text'}
                        secondary={`Confidence: ${(result.confidence * 100).toFixed(1)}%`}
                      />
                    </ListItem>
                  )) || <Typography color="textSecondary">No text detected</Typography>}
                </List>
              </TabPanel>
              
              {/* Stats Tab */}
              <TabPanel value={tabValue} index={3}>
                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                  <Typography variant="body2">
                    Frames Processed: {stats.frames_processed}
                  </Typography>
                  <Typography variant="body2">
                    Total Detections: {stats.total_detections}
                  </Typography>
                  <Typography variant="body2">
                    Total OCR Results: {stats.total_ocr_results}
                  </Typography>
                  <Typography variant="body2">
                    Avg Processing Time: {stats.avg_processing_time_ms?.toFixed(2)}ms
                  </Typography>
                  <Typography variant="body2">
                    Session Duration: {Math.floor(stats.session_duration_s || 0)}s
                  </Typography>
                  
                  {/* Active Sessions */}
                  <Typography variant="h6" sx={{ mt: 2 }}>
                    Active Sessions
                  </Typography>
                  {Object.keys(sessions).length > 0 ? (
                    Object.entries(sessions).map(([id, session]) => (
                      <Chip
                        key={id}
                        label={`${id} (${session.frame_count} frames)`}
                        onClick={() => setSessionId(id)}
                        variant={id === sessionId ? "filled" : "outlined"}
                        size="small"
                        sx={{ mr: 1, mb: 1 }}
                      />
                    ))
                  ) : (
                    <Typography color="textSecondary">No active sessions</Typography>
                  )}
                </Box>
              </TabPanel>
            </CardContent>
          </Card>
          
          {/* Frame History */}
          <Card sx={{ mt: 2 }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Frame History
              </Typography>
              <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                {frameHistory.slice(-10).map((frame, index) => (
                  <Tooltip key={frame.frame_id} title={`Frame ${frame.frame_id}`}>
                    <IconButton
                      size="small"
                      onClick={() => updateCanvas(frame.image)}
                      sx={{
                        border: currentFrame?.frame_id === frame.frame_id ? '2px solid blue' : '1px solid gray',
                        borderRadius: 1
                      }}
                    >
                      <img
                        src={`data:image/jpeg;base64,${frame.image}`}
                        alt={`Frame ${frame.frame_id}`}
                        style={{ width: 40, height: 30, objectFit: 'cover' }}
                      />
                    </IconButton>
                  </Tooltip>
                ))}
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
}