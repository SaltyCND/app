import React, { useState, useRef } from 'react';
import { useAppSelector, useAppDispatch } from '../app/hooks';
import { addMessage, updateLastMessage, ChatMessage } from '../features/chat/chatSlice';
import { Box, Button, TextField, Paper, Typography, Avatar, Stack } from '@mui/material';
import { motion } from 'framer-motion';
import ReactMarkdown from 'react-markdown';
import AccountCircleIcon from '@mui/icons-material/AccountCircle';
import ElectricBoltIcon from '@mui/icons-material/ElectricBolt';
import TypingIndicator from './TypingIndicator';

const ChatInterface: React.FC = () => {
  const conversation = useAppSelector((state) => state.chat.conversation);
  const dispatch = useAppDispatch();

  const [input, setInput] = useState<string>('');
  const [streaming, setStreaming] = useState<boolean>(false);
  const liveAnswerRef = useRef<string>("");

  const startVoiceInput = () => {
    const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    if (!SpeechRecognition) {
      alert("Your browser does not support speech recognition.");
      return;
    }
    const recognition = new SpeechRecognition();
    recognition.lang = 'en-US';
    recognition.start();
    recognition.onresult = (event: any) => {
      const transcript = event.results[0][0].transcript;
      setInput(transcript);
    };
  };

  const speakText = (text: string) => {
    const utterance = new SpeechSynthesisUtterance(text);
    window.speechSynthesis.speak(utterance);
  };

  const handleSend = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (!input.trim() || streaming) return;

    // Add user message
    dispatch(addMessage({ role: 'user', content: input }));
    
    // Add a single empty assistant message that we'll update
    dispatch(addMessage({ role: 'assistant', content: '' }));
    
    setStreaming(true);
    liveAnswerRef.current = "";

    console.log("Opening SSE connection for query:", input);
    const eventSource = new EventSource(
      `http://localhost:8000/query/stream?query=${encodeURIComponent(input)}`
    );

    eventSource.onmessage = (event) => {
      if (event.data.trim() === "[DONE]") {
        eventSource.close();
        setStreaming(false);
      } else {
        liveAnswerRef.current += event.data + " ";
        dispatch(updateLastMessage(liveAnswerRef.current.trim()));
      }
    };

    eventSource.onerror = (err) => {
      console.error("SSE error:", err);
      eventSource.close();
      setStreaming(false);
    };

    setInput('');
  };

  return (
    <Box sx={{ mt: 4 }}>
      <Typography variant="h5" gutterBottom>Chat with Electrical AI</Typography>
      <Paper sx={{ p: 2, mb: 2, maxHeight: 400, overflowY: 'auto' }}>
        {conversation.map((msg: ChatMessage, index: number) => (
          <motion.div
            key={index}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, ease: "easeOut" }}
          >
            <Stack direction="row" spacing={1} alignItems="flex-start" sx={{ mb: 1 }}>
              <Avatar sx={{ bgcolor: msg.role === 'user' ? 'primary.main' : 'secondary.main' }}>
                {msg.role === 'user' ? <AccountCircleIcon /> : <ElectricBoltIcon />}
              </Avatar>
              <Box
                sx={{
                  p: 1.5,
                  backgroundColor: msg.role === 'user' ? '#dcf8c6' : '#f1f0f0',
                  borderRadius: 2,
                  boxShadow: 1,
                  maxWidth: '80%',
                }}
              >
                <ReactMarkdown>{msg.content}</ReactMarkdown>
                {streaming && index === conversation.length - 1 && msg.role === 'assistant' && (
                  <TypingIndicator />
                )}
              </Box>
            </Stack>
          </motion.div>
        ))}
      </Paper>
      <Box component="form" onSubmit={handleSend} sx={{ display: 'flex', gap: 1 }}>
        <TextField
          fullWidth
          variant="outlined"
          placeholder="Type your message..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          disabled={streaming}
        />
        <Button type="submit" variant="contained" disabled={streaming}>
          {streaming ? 'Streaming...' : 'Send'}
        </Button>
        <Button variant="outlined" onClick={startVoiceInput} disabled={streaming}>
          🎤 Voice
        </Button>
      </Box>
    </Box>
  );
};

export default ChatInterface;