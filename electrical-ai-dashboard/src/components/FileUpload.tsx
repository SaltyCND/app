// src/components/FileUpload.tsx
import React, { useState } from 'react';
import { Box, Button, Typography, Paper } from '@mui/material';
import axios from 'axios';

interface FileUploadProps {
  endpoint: string; // e.g., "upload_receipt"
  label: string;
}

const FileUpload: React.FC<FileUploadProps> = ({ endpoint, label }) => {
  const [file, setFile] = useState<File | null>(null);
  const [responseMessage, setResponseMessage] = useState<string>('');

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      setFile(e.target.files[0]);
    }
  };

  const handleUpload = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (!file) return;
    const formData = new FormData();
    formData.append('file', file);
    try {
      const response = await axios.post<{ result: string }>(
        `http://localhost:8000/${endpoint}`,
        formData,
        { headers: { 'Content-Type': 'multipart/form-data' } }
      );
      setResponseMessage(response.data.result);
    } catch (error) {
      console.error('Error uploading file:', error);
      setResponseMessage('Error processing the file.');
    }
  };

  return (
    <Box sx={{ mt: 4 }}>
      <Typography variant="h5" gutterBottom>
        {label}
      </Typography>
      <Paper sx={{ p: 2 }}>
        <Box component="form" onSubmit={handleUpload} sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
          <input type="file" onChange={handleFileChange} />
          <Button type="submit" variant="contained">
            Upload
          </Button>
        </Box>
        {responseMessage && (
          <Box sx={{ mt: 2 }}>
            <Typography variant="subtitle1">Response:</Typography>
            <Typography variant="body2" sx={{ whiteSpace: 'pre-wrap' }}>
              {responseMessage}
            </Typography>
          </Box>
        )}
      </Paper>
    </Box>
  );
};

export default FileUpload;