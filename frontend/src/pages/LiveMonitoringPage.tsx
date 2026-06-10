import React from 'react';
import { LiveMonitoring } from '../features/live-monitoring/components/LiveMonitoring';

export default function LiveMonitoringPage({ lastIncidentEvent = 0 }: { lastIncidentEvent?: number }) {
  return <LiveMonitoring lastIncidentEvent={lastIncidentEvent} />;
}