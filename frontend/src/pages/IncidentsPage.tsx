import React from 'react';
import { IncidentsList } from '../features/incidents/components/IncidentsList';

export default function IncidentsPage({ query = '' }: { query?: string }) {
  return <IncidentsList query={query} />;
}

