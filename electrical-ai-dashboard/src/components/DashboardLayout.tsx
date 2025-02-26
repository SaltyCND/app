// src/components/DashboardLayout.tsx
import React from 'react';
import { Box, Drawer, List, ListItem, ListItemIcon, ListItemText, AppBar, Toolbar, Typography, CssBaseline, IconButton } from '@mui/material';
import { NavLink } from 'react-router-dom';
import ChatIcon from '@mui/icons-material/Chat';
import MenuBookIcon from '@mui/icons-material/MenuBook';
import ReceiptIcon from '@mui/icons-material/Receipt';
import AssessmentIcon from '@mui/icons-material/Assessment';
import QuizIcon from '@mui/icons-material/Quiz';
import AutoGraphIcon from '@mui/icons-material/AutoGraph';
import MenuIcon from '@mui/icons-material/Menu';

const drawerWidth = 240;

const navItems = [
  { text: 'Chat', icon: <ChatIcon />, route: '/chat' },
  { text: 'Electrical Wiki', icon: <MenuBookIcon />, route: '/wiki' },
  { text: 'Receipts', icon: <ReceiptIcon />, route: '/receipts' },
  { text: 'Reports', icon: <AssessmentIcon />, route: '/reports' },
  { text: 'Training', icon: <QuizIcon />, route: '/training' },
  { text: 'Diagrams', icon: <AutoGraphIcon />, route: '/diagram' },
];

const DashboardLayout: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  return (
    <Box sx={{ display: 'flex' }}>
      <CssBaseline />
      {/* App Bar with logo and menu icon */}
      <AppBar position="fixed" sx={{ width: `calc(100% - ${drawerWidth}px)`, ml: `${drawerWidth}px` }}>
        <Toolbar>
          {/* Example Logo */}
          <Box
            component="img"
            src="/ai-logo.png" // Replace with your logo path
            alt="Company Logo"
            sx={{ height: 40, mr: 2 }}
          />
          <Typography variant="h6" noWrap component="div" sx={{ flexGrow: 1 }}>
            Electrical AI Dashboard
          </Typography>
          {/* Optional: Add a search field or profile icon here */}
          <IconButton color="inherit">
            <MenuIcon />
          </IconButton>
        </Toolbar>
      </AppBar>
      {/* Sidebar Navigation */}
      <Drawer
        variant="permanent"
        sx={{
          width: drawerWidth,
          flexShrink: 0,
          [`& .MuiDrawer-paper`]: { width: drawerWidth, boxSizing: 'border-box' },
        }}
      >
        <Toolbar />
        <Box sx={{ overflow: 'auto' }}>
          <List>
            {navItems.map((item, index) => (
              <ListItem key={index} disablePadding>
                <NavLink
                  to={item.route}
                  style={({ isActive }: { isActive: boolean }) => ({
                    display: 'flex',
                    alignItems: 'center',
                    width: '100%',
                    padding: '8px 16px',
                    backgroundColor: isActive ? '#e0e0e0' : 'inherit',
                    textDecoration: 'none',
                    color: 'inherit',
                  })}
                >
                  <ListItemIcon>{item.icon}</ListItemIcon>
                  <ListItemText primary={item.text} />
                </NavLink>
              </ListItem>
            ))}
          </List>
        </Box>
      </Drawer>
      {/* Main Content */}
      <Box component="main" sx={{ flexGrow: 1, bgcolor: 'background.default', p: 3 }}>
        <Toolbar />
        {children}
      </Box>
    </Box>
  );
};

export default DashboardLayout;