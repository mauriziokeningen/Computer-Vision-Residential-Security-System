import React from 'react';
import { Dashboard } from '../features/alerts/components/Dashboard';
import { AlertCounts } from '../types';

export default function DashboardPage({ alertCounts }: { alertCounts: AlertCounts }) {
  return <Dashboard alertCounts={alertCounts} />;
}

