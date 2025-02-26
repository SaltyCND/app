import React, { Suspense, lazy } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import DashboardLayout from './components/DashboardLayout';

const ChatInterface = lazy(() => import('./components/ChatInterface'));
const Wiki = lazy(() => import('./components/Wiki'));
const Reports = lazy(() => import('./components/Reports'));
const Receipts = lazy(() => import('./components/Receipts'));
const Training = lazy(() => import('./components/Training'));
const Diagram = lazy(() => import('./components/Diagram'));

const App: React.FC = () => {
  return (
    <Router>
      <DashboardLayout>
        <Suspense fallback={<div>Loading...</div>}>
          <Routes>
            <Route path="/chat" element={<ChatInterface />} />
            <Route path="/wiki" element={<Wiki />} />
            <Route path="/reports" element={<Reports />} />
            <Route path="/receipts" element={<Receipts />} />
            <Route path="/training" element={<Training />} />
            <Route path="/diagram" element={<Diagram />} />
            <Route path="/" element={<Navigate to="/chat" replace />} />
          </Routes>
        </Suspense>
      </DashboardLayout>
    </Router>
  );
};

export default App;