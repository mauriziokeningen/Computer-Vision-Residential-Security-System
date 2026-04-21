import React from 'react';
import { Alerts } from '../features/alerts/components/Alerts';

export default function AlertsPage({ query = '' }: { query?: string }) {
  return <Alerts query={query} />;
}

