// src/components/Diagram.tsx
import React, { useState } from 'react';
import { Box, Paper, Typography, Button, TextField } from '@mui/material';
import axios from 'axios';

const Diagram: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [analysis, setAnalysis] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(false);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
    }
  };

  const handleAnalyze = async () => {
    if (!file) return;
    setLoading(true);
    const formData = new FormData();
    formData.append('file', file);
    try {
      const response = await axios.post('http://localhost:8000/diagram', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setAnalysis(response.data.result);
    } catch (error: any) {
      console.error('Error analyzing diagram:', error);
      setAnalysis('Error analyzing diagram.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box sx={{ p: { xs: 2, sm: 3 } }}>
      <Paper sx={{ p: 3 }}>
        <Typography variant="h4" gutterBottom>Diagram Analysis</Typography>
        <Typography variant="body1" sx={{ mb: 2 }}>
          Upload a technical diagram for automated analysis and troubleshooting recommendations.
        </Typography>
        <TextField type="file" onChange={handleFileChange} fullWidth sx={{ mb: 2 }} />
        <Button variant="contained" onClick={handleAnalyze} disabled={loading}>
          {loading ? 'Analyzing...' : 'Upload Diagram'}
        </Button>
        {analysis && (
          <Typography variant="body2" sx={{ mt: 2 }}>
            {analysis}
          </Typography>
        )}
      </Paper>
    </Box>
  );
};

export default Diagram;