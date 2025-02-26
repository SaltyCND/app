// src/components/Notification.tsx
import React, { useState } from 'react';
import { Snackbar, Alert } from '@mui/material';

interface NotificationProps {
  message: string;
  severity?: "success" | "info" | "warning" | "error";
}

const Notification: React.FC<NotificationProps> = ({ message, severity = "info" }) => {
  const [open, setOpen] = useState<boolean>(true);

  const handleClose = () => {
    setOpen(false);
  };

  return (
    <Snackbar open={open} autoHideDuration={3000} onClose={handleClose}>
      <Alert onClose={handleClose} severity={severity} sx={{ width: '100%' }}>
        {message}
      </Alert>
    </Snackbar>
  );
};

export default Notification;