// src/components/Reports.tsx
import React, { useState } from 'react';
import { Box, Paper, Typography, TextField, Button, Skeleton } from '@mui/material';
import axios from 'axios';

const Reports: React.FC = () => {
  const [parameters, setParameters] = useState<string>('');
  const [report, setReport] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(false);

  const handleGenerateReport = async () => {
    if (!parameters.trim()) return;
    setLoading(true);
    try {
      const response = await axios.post<{ answer: string }>('http://localhost:8000/report', { query: parameters });
      setReport(response.data.answer);
    } catch (error: any) {
      console.error('Error generating report:', error);
      setReport('Error generating report.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box sx={{ p: { xs: 2, sm: 3 } }}>
      <Paper sx={{ p: 3 }}>
        <Typography variant="h4" gutterBottom>Reports</Typography>
        <TextField
          fullWidth
          variant="outlined"
          placeholder="Enter report parameters..."
          value={parameters}
          onChange={(e) => setParameters(e.target.value)}
          sx={{ mb: 2 }}
        />
        <Button variant="contained" onClick={handleGenerateReport} disabled={loading}>
          {loading ? 'Generating Report...' : 'Generate Report'}
        </Button>
        {loading ? (
          <Skeleton variant="rectangular" width="100%" height={200} sx={{ mt: 2 }} />
        ) : (
          report && (
            <Typography variant="body1" sx={{ mt: 2 }}>
              {report}
            </Typography>
          )
        )}
      </Paper>
    </Box>
  );
};

export default Reports;