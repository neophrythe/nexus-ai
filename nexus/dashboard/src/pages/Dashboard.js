import React, { useState, useEffect } from 'react';
import {
  Grid,
  Paper,
  Typography,
  Box,
  Card,
  CardContent,
  LinearProgress,
  Chip,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  IconButton,
} from '@mui/material';
import {
  Speed,
  Memory,
  Storage,
  Videocam,
  Psychology,
  Games,
  PlayArrow,
  Stop,
  Refresh,
} from '@mui/icons-material';
import { Line, Doughnut } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  ArcElement,
} from 'chart.js';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  ArcElement
);

function Dashboard({ ws }) {
  const [performance, setPerformance] = useState({
    cpu: 0,
    memory: 0,
    gpu: 0,
    fps: 0,
    captureTime: 0,
  });

  const [agents, setAgents] = useState([]);
  const [environments, setEnvironments] = useState([]);
  const [recentEvents, setRecentEvents] = useState([]);
  const [captureStats, setCaptureStats] = useState(null);

  useEffect(() => {
    fetchDashboardData();
    const interval = setInterval(fetchDashboardData, 2000);
    return () => clearInterval(interval);
  }, []);

  const fetchDashboardData = async () => {
    try {
      // Fetch performance metrics
      const perfResponse = await fetch('/performance');
      const perfData = await perfResponse.json();
      setPerformance({
        cpu: perfData.cpu_percent || 0,
        memory: perfData.memory_mb || 0,
        gpu: 0, // TODO: Add GPU monitoring
        fps: perfData.capture_stats?.current_fps || 0,
        captureTime: perfData.capture_stats?.avg_capture_time || 0,
      });
      setCaptureStats(perfData.capture_stats);

      // Fetch agents
      const agentsResponse = await fetch('/agents');
      const agentsData = await agentsResponse.json();
      setAgents(agentsData.agents || []);

      // Fetch plugins (environments)
      const pluginsResponse = await fetch('/plugins');
      const pluginsData = await pluginsResponse.json();
      setEnvironments(pluginsData.loaded || {});
    } catch (error) {
      console.error('Failed to fetch dashboard data:', error);
    }
  };

  const performanceChartData = {
    labels: ['CPU', 'Memory', 'GPU'],
    datasets: [
      {
        data: [performance.cpu, performance.memory / 100, performance.gpu],
        backgroundColor: ['#00ff41', '#ff00ff', '#00ffff'],
        borderColor: ['#00ff41', '#ff00ff', '#00ffff'],
        borderWidth: 1,
      },
    ],
  };

  const fpsChartData = {
    labels: Array.from({ length: 20 }, (_, i) => i),
    datasets: [
      {
        label: 'FPS',
        data: Array(20).fill(performance.fps),
        borderColor: '#00ff41',
        backgroundColor: 'rgba(0, 255, 65, 0.2)',
        tension: 0.4,
      },
    ],
  };

  const MetricCard = ({ title, value, unit, icon, color }) => (
    <Card sx={{ height: '100%', bgcolor: 'background.paper' }}>
      <CardContent>
        <Box display="flex" alignItems="center" mb={2}>
          <Box sx={{ color: color || 'primary.main', mr: 2 }}>
            {icon}
          </Box>
          <Typography variant="h6">{title}</Typography>
        </Box>
        <Typography variant="h3" sx={{ color: color || 'primary.main' }}>
          {value}
          <Typography component="span" variant="h6" sx={{ ml: 1 }}>
            {unit}
          </Typography>
        </Typography>
      </CardContent>
    </Card>
  );

  return (
    <Box>
      <Typography variant="h4" gutterBottom sx={{ color: 'primary.main' }}>
        System Dashboard
      </Typography>

      <Grid container spacing={3}>
        {/* Metric Cards */}
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="CPU Usage"
            value={performance.cpu.toFixed(1)}
            unit="%"
            icon={<Speed />}
            color="#00ff41"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="Memory"
            value={performance.memory.toFixed(0)}
            unit="MB"
            icon={<Memory />}
            color="#ff00ff"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="FPS"
            value={performance.fps.toFixed(1)}
            unit="fps"
            icon={<Videocam />}
            color="#00ffff"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="Capture Time"
            value={performance.captureTime.toFixed(2)}
            unit="ms"
            icon={<Speed />}
            color="#ffff00"
          />
        </Grid>

        {/* Performance Charts */}
        <Grid item xs={12} md={8}>
          <Paper sx={{ p: 2, height: 400 }}>
            <Typography variant="h6" gutterBottom>
              FPS History
            </Typography>
            <Box sx={{ height: 320 }}>
              <Line
                data={fpsChartData}
                options={{
                  responsive: true,
                  maintainAspectRatio: false,
                  scales: {
                    y: {
                      beginAtZero: true,
                      max: 120,
                      grid: {
                        color: 'rgba(255, 255, 255, 0.1)',
                      },
                    },
                    x: {
                      grid: {
                        color: 'rgba(255, 255, 255, 0.1)',
                      },
                    },
                  },
                  plugins: {
                    legend: {
                      display: false,
                    },
                  },
                }}
              />
            </Box>
          </Paper>
        </Grid>

        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 2, height: 400 }}>
            <Typography variant="h6" gutterBottom>
              Resource Usage
            </Typography>
            <Box sx={{ height: 320, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <Doughnut
                data={performanceChartData}
                options={{
                  responsive: true,
                  maintainAspectRatio: false,
                  plugins: {
                    legend: {
                      position: 'bottom',
                    },
                  },
                }}
              />
            </Box>
          </Paper>
        </Grid>

        {/* Active Components */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2, height: 400 }}>
            <Typography variant="h6" gutterBottom>
              Active Agents
            </Typography>
            <List>
              {agents.length > 0 ? (
                agents.map((agent, index) => (
                  <ListItem key={index}>
                    <ListItemIcon>
                      <Psychology sx={{ color: 'primary.main' }} />
                    </ListItemIcon>
                    <ListItemText
                      primary={agent.name}
                      secondary={`Type: ${agent.type} | Status: ${agent.status}`}
                    />
                    <IconButton size="small">
                      <PlayArrow />
                    </IconButton>
                    <IconButton size="small">
                      <Stop />
                    </IconButton>
                  </ListItem>
                ))
              ) : (
                <Typography variant="body2" sx={{ p: 2, color: 'text.secondary' }}>
                  No active agents
                </Typography>
              )}
            </List>
          </Paper>
        </Grid>

        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2, height: 400 }}>
            <Typography variant="h6" gutterBottom>
              Loaded Plugins
            </Typography>
            <List>
              {Object.keys(environments).length > 0 ? (
                Object.entries(environments).map(([name, info]) => (
                  <ListItem key={name}>
                    <ListItemIcon>
                      <Games sx={{ color: 'secondary.main' }} />
                    </ListItemIcon>
                    <ListItemText
                      primary={name}
                      secondary={`Status: ${info.status}`}
                    />
                    <Chip
                      label={info.type}
                      size="small"
                      color="primary"
                      variant="outlined"
                    />
                  </ListItem>
                ))
              ) : (
                <Typography variant="body2" sx={{ p: 2, color: 'text.secondary' }}>
                  No plugins loaded
                </Typography>
              )}
            </List>
          </Paper>
        </Grid>

        {/* Capture Statistics */}
        {captureStats && (
          <Grid item xs={12}>
            <Paper sx={{ p: 2 }}>
              <Typography variant="h6" gutterBottom>
                Capture Statistics
              </Typography>
              <Grid container spacing={2}>
                <Grid item xs={6} md={3}>
                  <Typography variant="body2" color="text.secondary">
                    Total Frames
                  </Typography>
                  <Typography variant="h6">
                    {captureStats.total_frames}
                  </Typography>
                </Grid>
                <Grid item xs={6} md={3}>
                  <Typography variant="body2" color="text.secondary">
                    Dropped Frames
                  </Typography>
                  <Typography variant="h6" sx={{ color: captureStats.dropped_frames > 0 ? 'error.main' : 'success.main' }}>
                    {captureStats.dropped_frames}
                  </Typography>
                </Grid>
                <Grid item xs={6} md={3}>
                  <Typography variant="body2" color="text.secondary">
                    Min Capture Time
                  </Typography>
                  <Typography variant="h6">
                    {captureStats.min_capture_time?.toFixed(2)} ms
                  </Typography>
                </Grid>
                <Grid item xs={6} md={3}>
                  <Typography variant="body2" color="text.secondary">
                    Max Capture Time
                  </Typography>
                  <Typography variant="h6">
                    {captureStats.max_capture_time?.toFixed(2)} ms
                  </Typography>
                </Grid>
              </Grid>
            </Paper>
          </Grid>
        )}
      </Grid>
    </Box>
  );
}

export default Dashboard;