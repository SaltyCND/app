// src/components/Wiki.tsx
import React, { useState, useEffect } from 'react';
import { Box, Paper, Typography, Skeleton } from '@mui/material';

const Wiki: React.FC = () => {
  const [loading, setLoading] = useState<boolean>(true);
  const [content, setContent] = useState<string>("");

  useEffect(() => {
    // Simulate an API call to fetch wiki content
    setTimeout(() => {
      setContent("Here you will find detailed manuals, troubleshooting guides, and best practices for electrical equipment.");
      setLoading(false);
    }, 1500);
  }, []);

  return (
    <Box sx={{ p: { xs: 2, sm: 3 } }}>
      <Paper sx={{ p: 3 }}>
        <Typography variant="h4" gutterBottom>Electrical Wiki</Typography>
        {loading ? (
          <Skeleton variant="rectangular" width="100%" height={150} />
        ) : (
          <Typography variant="body1">{content}</Typography>
        )}
      </Paper>
    </Box>
  );
};

export default Wiki;