import React from 'react';
import {
  AppBar,
  Toolbar,
  IconButton,
  Typography,
  Box,
  Chip
} from '@mui/material';
import {
  Menu as MenuIcon,
  Circle
} from '@mui/icons-material';

export default function TopBar({ onMenuClick, systemStatus }) {
  const getStatusColor = () => {
    if (!systemStatus) return 'default';
    return systemStatus.status === 'online' ? 'success' : 'error';
  };

  const getStatusText = () => {
    if (!systemStatus) return 'Unknown';
    return systemStatus.status === 'online' ? 'Online' : 'Offline';
  };

  return (
    <AppBar
      position="fixed"
      sx={{
        zIndex: (theme) => theme.zIndex.drawer + 1,
        backgroundColor: '#000',
        borderBottom: '1px solid #333',
      }}
    >
      <Toolbar>
        <IconButton
          color="inherit"
          aria-label="toggle drawer"
          onClick={onMenuClick}
          edge="start"
          sx={{ mr: 2 }}
        >
          <MenuIcon />
        </IconButton>

        <Typography
          variant="h6"
          noWrap
          component="div"
          sx={{
            flexGrow: 1,
            color: '#00ff41',
            fontWeight: 'bold',
          }}
        >
          Nexus Game Automation Framework
        </Typography>

        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          {/* System Status */}
          <Chip
            icon={<Circle sx={{ fontSize: 12 }} />}
            label={getStatusText()}
            color={getStatusColor()}
            size="small"
            variant="outlined"
          />

          {/* Component Status */}
          {systemStatus?.components && (
            <Box sx={{ display: 'flex', gap: 1 }}>
              {systemStatus.components.capture_manager && (
                <Chip
                  label="Capture"
                  size="small"
                  color="success"
                  variant="outlined"
                />
              )}
              {systemStatus.components.agent && (
                <Chip
                  label="Agent"
                  size="small"
                  color="success"
                  variant="outlined"
                />
              )}
              {systemStatus.components.stream && (
                <Chip
                  label="Stream"
                  size="small"
                  color="info"
                  variant="outlined"
                />
              )}
            </Box>
          )}

          {/* Timestamp */}
          {systemStatus?.timestamp && (
            <Typography
              variant="caption"
              sx={{ color: '#666', ml: 2 }}
            >
              {new Date(systemStatus.timestamp).toLocaleTimeString()}
            </Typography>
          )}
        </Box>
      </Toolbar>
    </AppBar>
  );
}