import React from 'react';
import { IncidentsList } from '../features/incidents/components/IncidentsList';

export default function IncidentsPage({ query = '', lastIncidentEvent = 0 }: { query?: string; lastIncidentEvent?: number }) {
  return <IncidentsList query={query} lastIncidentEvent={lastIncidentEvent} />;
}