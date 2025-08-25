import React, { useState, useEffect, useRef } from 'react';
import {
  Box,
  Paper,
  Typography,
  Button,
  Grid,
  Card,
  CardContent,
  IconButton,
  Slider,
  Switch,
  FormControlLabel,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Chip,
  ToggleButton,
  ToggleButtonGroup,
} from '@mui/material';
import {
  PlayArrow,
  Stop,
  Refresh,
  Fullscreen,
  Settings,
  CropFree,
  Visibility,
  VisibilityOff,
  Screenshot,
  FiberManualRecord,
} from '@mui/icons-material';

function LiveView({ ws }) {
  const canvasRef = useRef(null);
  const [streaming, setStreaming] = useState(false);
  const [fps, setFps] = useState(30);
  const [showOverlay, setShowOverlay] = useState(true);
  const [detectionEnabled, setDetectionEnabled] = useState(true);
  const [ocrEnabled, setOcrEnabled] = useState(false);
  const [currentFrame, setCurrentFrame] = useState(null);
  const [frameStats, setFrameStats] = useState({
    frameId: 0,
    captureTime: 0,
    objects: [],
    text: [],
  });
  const [captureRegion, setCaptureRegion] = useState(null);
  const [recording, setRecording] = useState(false);

  useEffect(() => {
    if (ws) {
      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === 'stream_frame') {
          handleStreamFrame(data);
        }
      };
    }
  }, [ws]);

  const handleStreamFrame = (data) => {
    if (canvasRef.current) {
      const ctx = canvasRef.current.getContext('2d');
      const img = new Image();
      img.onload = () => {
        ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
        ctx.drawImage(img, 0, 0, canvasRef.current.width, canvasRef.current.height);
        
        // Draw overlays if enabled
        if (showOverlay && data.detections) {
          drawOverlays(ctx, data.detections);
        }
      };
      img.src = `data:image/jpeg;base64,${data.data}`;
      
      setCurrentFrame(data);
      setFrameStats({
        frameId: data.frame_id,
        captureTime: data.capture_time_ms || 0,
        objects: data.detections?.objects || [],
        text: data.detections?.text || [],
      });
    }
  };

  const drawOverlays = (ctx, detections) => {
    // Draw object detections
    if (detections.objects) {
      ctx.strokeStyle = '#00ff41';
      ctx.lineWidth = 2;
      ctx.font = '14px Roboto Mono';
      
      detections.objects.forEach(obj => {
        const [x1, y1, x2, y2] = obj.bbox;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
        
        // Draw label
        ctx.fillStyle = 'rgba(0, 255, 65, 0.8)';
        ctx.fillRect(x1, y1 - 20, 150, 20);
        ctx.fillStyle = 'black';
        ctx.fillText(`${obj.class}: ${obj.confidence.toFixed(2)}`, x1 + 5, y1 - 5);
      });
    }
    
    // Draw text detections
    if (detections.text) {
      ctx.strokeStyle = '#ff00ff';
      ctx.lineWidth = 2;
      
      detections.text.forEach(txt => {
        const [x1, y1, x2, y2] = txt.bbox;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
        
        // Draw text label
        ctx.fillStyle = 'rgba(255, 0, 255, 0.8)';
        ctx.fillRect(x1, y2, txt.text.length * 8, 20);
        ctx.fillStyle = 'white';
        ctx.fillText(txt.text, x1 + 5, y2 + 15);
      });
    }
  };

  const startStream = async () => {
    try {
      const response = await fetch('/capture/stream/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ fps }),
      });
      const data = await response.json();
      setStreaming(true);
    } catch (error) {
      console.error('Failed to start stream:', error);
    }
  };

  const stopStream = async () => {
    try {
      await fetch('/capture/stream/stop', { method: 'POST' });
      setStreaming(false);
    } catch (error) {
      console.error('Failed to stop stream:', error);
    }
  };

  const captureFrame = async () => {
    try {
      const response = await fetch('/capture/frame', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ region: captureRegion }),
      });
      const data = await response.json();
      
      // Download the frame
      const link = document.createElement('a');
      link.download = `capture_${data.frame_id}.jpg`;
      link.href = `data:image/jpeg;base64,${data.image}`;
      link.click();
    } catch (error) {
      console.error('Failed to capture frame:', error);
    }
  };

  const toggleRecording = () => {
    setRecording(!recording);
    // TODO: Implement recording functionality
  };

  const setRegionOfInterest = () => {
    // TODO: Implement ROI selection
    setCaptureRegion([100, 100, 800, 600]);
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom sx={{ color: 'primary.main' }}>
        Live Game View
      </Typography>

      <Grid container spacing={3}>
        {/* Main View */}
        <Grid item xs={12} lg={9}>
          <Paper sx={{ p: 2, position: 'relative' }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
              <Box>
                <Button
                  variant="contained"
                  startIcon={streaming ? <Stop /> : <PlayArrow />}
                  onClick={streaming ? stopStream : startStream}
                  sx={{ mr: 2 }}
                >
                  {streaming ? 'Stop' : 'Start'} Stream
                </Button>
                <IconButton onClick={captureFrame}>
                  <Screenshot />
                </IconButton>
                <IconButton onClick={toggleRecording} color={recording ? 'error' : 'default'}>
                  <FiberManualRecord />
                </IconButton>
                <IconButton onClick={setRegionOfInterest}>
                  <CropFree />
                </IconButton>
                <IconButton>
                  <Fullscreen />
                </IconButton>
              </Box>
              
              <Box>
                <FormControlLabel
                  control={
                    <Switch
                      checked={showOverlay}
                      onChange={(e) => setShowOverlay(e.target.checked)}
                    />
                  }
                  label="Overlay"
                />
                <IconButton>
                  <Settings />
                </IconButton>
              </Box>
            </Box>

            <Box
              sx={{
                width: '100%',
                height: 600,
                backgroundColor: 'black',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                position: 'relative',
              }}
            >
              <canvas
                ref={canvasRef}
                width={1280}
                height={720}
                style={{
                  maxWidth: '100%',
                  maxHeight: '100%',
                  objectFit: 'contain',
                }}
              />
              
              {!streaming && (
                <Typography
                  variant="h6"
                  sx={{
                    position: 'absolute',
                    color: 'rgba(255, 255, 255, 0.5)',
                  }}
                >
                  Stream not active
                </Typography>
              )}
              
              {captureRegion && (
                <Box
                  sx={{
                    position: 'absolute',
                    border: '2px dashed #00ff41',
                    left: `${captureRegion[0] / 10}%`,
                    top: `${captureRegion[1] / 10}%`,
                    width: `${(captureRegion[2] - captureRegion[0]) / 10}%`,
                    height: `${(captureRegion[3] - captureRegion[1]) / 10}%`,
                    pointerEvents: 'none',
                  }}
                />
              )}
            </Box>

            {/* Stream Controls */}
            <Box sx={{ mt: 2 }}>
              <Grid container spacing={2} alignItems="center">
                <Grid item xs={4}>
                  <Typography gutterBottom>FPS: {fps}</Typography>
                  <Slider
                    value={fps}
                    onChange={(e, value) => setFps(value)}
                    min={1}
                    max={120}
                    marks={[
                      { value: 30, label: '30' },
                      { value: 60, label: '60' },
                      { value: 120, label: '120' },
                    ]}
                  />
                </Grid>
                <Grid item xs={4}>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={detectionEnabled}
                        onChange={(e) => setDetectionEnabled(e.target.checked)}
                      />
                    }
                    label="Object Detection"
                  />
                </Grid>
                <Grid item xs={4}>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={ocrEnabled}
                        onChange={(e) => setOcrEnabled(e.target.checked)}
                      />
                    }
                    label="Text Detection"
                  />
                </Grid>
              </Grid>
            </Box>
          </Paper>
        </Grid>

        {/* Side Panel */}
        <Grid item xs={12} lg={3}>
          {/* Frame Info */}
          <Card sx={{ mb: 2 }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Frame Info
              </Typography>
              <Typography variant="body2">
                Frame ID: {frameStats.frameId}
              </Typography>
              <Typography variant="body2">
                Capture Time: {frameStats.captureTime.toFixed(2)}ms
              </Typography>
              <Typography variant="body2">
                Objects: {frameStats.objects.length}
              </Typography>
              <Typography variant="body2">
                Text: {frameStats.text.length}
              </Typography>
            </CardContent>
          </Card>

          {/* Detected Objects */}
          <Card sx={{ mb: 2 }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Detected Objects
              </Typography>
              {frameStats.objects.length > 0 ? (
                <Box>
                  {frameStats.objects.map((obj, index) => (
                    <Chip
                      key={index}
                      label={`${obj.class} (${(obj.confidence * 100).toFixed(0)}%)`}
                      size="small"
                      sx={{ m: 0.5 }}
                      color="primary"
                      variant="outlined"
                    />
                  ))}
                </Box>
              ) : (
                <Typography variant="body2" color="text.secondary">
                  No objects detected
                </Typography>
              )}
            </CardContent>
          </Card>

          {/* Detected Text */}
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Detected Text
              </Typography>
              {frameStats.text.length > 0 ? (
                <Box>
                  {frameStats.text.map((txt, index) => (
                    <Typography key={index} variant="body2" sx={{ mb: 1 }}>
                      "{txt.text}" ({(txt.confidence * 100).toFixed(0)}%)
                    </Typography>
                  ))}
                </Box>
              ) : (
                <Typography variant="body2" color="text.secondary">
                  No text detected
                </Typography>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
}

export default LiveView;