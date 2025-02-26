// src/components/Receipts.tsx
import React, { useState } from 'react';
import { Box, Paper, Typography, Button, TextField } from '@mui/material';

const Receipts: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<string>("");
  const [loading, setLoading] = useState<boolean>(false);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
    }
  };

  const handleUpload = async () => {
    if (!file) return;
    setLoading(true);
    // Simulate file processing
    setTimeout(() => {
      setResult("Receipt processed successfully: Vendor XYZ, $123.45, Date: 2025-01-01");
      setLoading(false);
    }, 1500);
  };

  return (
    <Box sx={{ p: { xs: 2, sm: 3 } }}>
      <Paper sx={{ p: 3 }}>
        <Typography variant="h4" gutterBottom>Receipt Processing</Typography>
        <Typography variant="body1">
          Upload a receipt image to extract details automatically.
        </Typography>
        <Box sx={{ mt: 2 }}>
          <TextField type="file" onChange={handleFileChange} fullWidth />
          <Button variant="contained" sx={{ mt: 1 }} onClick={handleUpload} disabled={loading}>
            {loading ? "Processing..." : "Upload Receipt"}
          </Button>
        </Box>
        {result && (
          <Typography variant="body2" sx={{ mt: 2 }}>
            {result}
          </Typography>
        )}
      </Paper>
    </Box>
  );
};

export default Receipts;