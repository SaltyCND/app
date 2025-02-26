// src/components/Training.tsx
import React, { useState } from 'react';
import { Box, Paper, Typography, TextField, Button } from '@mui/material';
import axios from 'axios';

const Training: React.FC = () => {
  const [topic, setTopic] = useState<string>('');
  const [quiz, setQuiz] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(false);

  const handleGenerateQuiz = async () => {
    if (!topic.trim()) return;
    setLoading(true);
    try {
      const response = await axios.post<{ answer: string }>('http://localhost:8000/quiz', { query: topic });
      setQuiz(response.data.answer);
    } catch (error: any) {
      console.error('Error generating quiz:', error);
      setQuiz('Error generating quiz.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box sx={{ p: { xs: 2, sm: 3 } }}>
      <Paper sx={{ p: 3 }}>
        <Typography variant="h4" gutterBottom>Training & Assessment</Typography>
        <TextField
          fullWidth
          variant="outlined"
          placeholder="Enter training topic..."
          value={topic}
          onChange={(e) => setTopic(e.target.value)}
          sx={{ mb: 2 }}
        />
        <Button variant="contained" onClick={handleGenerateQuiz} disabled={loading}>
          {loading ? 'Generating Quiz...' : 'Generate Quiz'}
        </Button>
        {quiz && (
          <Typography variant="body1" sx={{ mt: 2 }}>
            {quiz}
          </Typography>
        )}
      </Paper>
    </Box>
  );
};

export default Training;