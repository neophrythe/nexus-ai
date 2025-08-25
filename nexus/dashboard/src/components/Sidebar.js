import React from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import {
  Drawer,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Divider,
  Typography,
  Box
} from '@mui/material';
import {
  Dashboard as DashboardIcon,
  Videocam,
  BugReport,
  Extension,
  School,
  Visibility,
  Analytics,
  SportsEsports,
  Settings
} from '@mui/icons-material';

const menuItems = [
  { text: 'Dashboard', icon: <DashboardIcon />, path: '/dashboard' },
  { text: 'Live View', icon: <Videocam />, path: '/live' },
  { text: 'Visual Debugger', icon: <BugReport />, path: '/debug' },
  { text: 'Plugins', icon: <Extension />, path: '/plugins' },
  { text: 'Training', icon: <School />, path: '/training' },
  { text: 'Vision', icon: <Visibility />, path: '/vision' },
  { text: 'Analytics', icon: <Analytics />, path: '/analytics' },
  { text: 'Game Control', icon: <SportsEsports />, path: '/game-control' },
  { text: 'Settings', icon: <Settings />, path: '/settings' },
];

export default function Sidebar({ open, onClose }) {
  const navigate = useNavigate();
  const location = useLocation();

  const handleItemClick = (path) => {
    navigate(path);
  };

  return (
    <Drawer
      variant="permanent"
      open={open}
      sx={{
        width: open ? 240 : 60,
        flexShrink: 0,
        [`& .MuiDrawer-paper`]: {
          width: open ? 240 : 60,
          boxSizing: 'border-box',
          backgroundColor: '#1a1a1a',
          borderRight: '1px solid #333',
          transition: 'width 0.3s',
          overflowX: 'hidden',
        },
      }}
    >
      <Box sx={{ mt: 8 }}>
        {/* Logo/Title */}
        <Box sx={{ p: 2, textAlign: 'center' }}>
          <Typography 
            variant="h6" 
            sx={{ 
              color: '#00ff41',
              fontWeight: 'bold',
              display: open ? 'block' : 'none'
            }}
          >
            NEXUS
          </Typography>
          {!open && (
            <Typography 
              variant="h4" 
              sx={{ 
                color: '#00ff41',
                fontWeight: 'bold'
              }}
            >
              N
            </Typography>
          )}
        </Box>
        
        <Divider sx={{ borderColor: '#333' }} />
        
        {/* Menu Items */}
        <List>
          {menuItems.map((item) => (
            <ListItem
              button
              key={item.text}
              onClick={() => handleItemClick(item.path)}
              selected={location.pathname === item.path}
              sx={{
                minHeight: 48,
                justifyContent: open ? 'initial' : 'center',
                px: 2.5,
                '&.Mui-selected': {
                  backgroundColor: '#00ff4120',
                  borderRight: '3px solid #00ff41',
                  '& .MuiListItemIcon-root': {
                    color: '#00ff41',
                  },
                  '& .MuiListItemText-primary': {
                    color: '#00ff41',
                  },
                },
                '&:hover': {
                  backgroundColor: '#333',
                },
              }}
            >
              <ListItemIcon
                sx={{
                  minWidth: 0,
                  mr: open ? 3 : 'auto',
                  justifyContent: 'center',
                  color: location.pathname === item.path ? '#00ff41' : '#fff',
                }}
              >
                {item.icon}
              </ListItemIcon>
              <ListItemText
                primary={item.text}
                sx={{
                  opacity: open ? 1 : 0,
                  '& .MuiListItemText-primary': {
                    fontSize: '0.9rem',
                    color: location.pathname === item.path ? '#00ff41' : '#fff',
                  },
                }}
              />
            </ListItem>
          ))}
        </List>
      </Box>
    </Drawer>
  );
}