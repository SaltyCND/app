// src/components/QueryForm.tsx
import React, { useState } from 'react';
import { Box, TextField, Button, Typography, Paper } from '@mui/material';
import axios from 'axios';

const QueryForm: React.FC = () => {
  const [query, setQuery] = useState<string>('');
  const [answer, setAnswer] = useState<string>('');

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    try {
      const response = await axios.post<{ answer: string }>('http://localhost:8000/query', { query });
      setAnswer(response.data.answer);
    } catch (error) {
      console.error('Error querying the AI:', error);
      setAnswer('Error processing your request.');
    }
  };

  return (
    <Box sx={{ mt: 4 }}>
      <Typography variant="h5" gutterBottom>
        Ask the Electrical AI
      </Typography>
      <Paper sx={{ p: 2 }}>
        <Box component="form" onSubmit={handleSubmit} sx={{ display: 'flex', gap: 1 }}>
          <TextField
            fullWidth
            variant="outlined"
            placeholder="Type your question here..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
          <Button type="submit" variant="contained">
            Submit
          </Button>
        </Box>
        {answer && (
          <Box sx={{ mt: 2 }}>
            <Typography variant="subtitle1">Answer:</Typography>
            <Typography variant="body1">{answer}</Typography>
          </Box>
        )}
      </Paper>
    </Box>
  );
};

export default QueryForm;