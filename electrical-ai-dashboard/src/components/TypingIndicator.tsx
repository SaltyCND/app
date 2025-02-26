import React from 'react';
import { Box, Typography, Stack } from '@mui/material';

const TypingIndicator: React.FC = () => {
  return (
    <Box sx={{ display: 'flex', alignItems: 'center', mt: 1 }}>
      <Typography variant="body2" color="textSecondary" sx={{ mr: 1 }}>
        AI is typing
      </Typography>
      <Stack direction="row" spacing={0.5}>
        {[...Array(3)].map((_, idx) => (
          <Box
            key={idx}
            sx={{
              width: 8,
              height: 8,
              borderRadius: '50%',
              bgcolor: 'primary.main',
              animation: 'blink 1.4s infinite both',
              animationDelay: `${idx * 0.2}s`,
            }}
          />
        ))}
      </Stack>
    </Box>
  );
};

export default TypingIndicator;