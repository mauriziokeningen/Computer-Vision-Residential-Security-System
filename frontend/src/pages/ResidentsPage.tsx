import React from 'react';
import { Residents } from '../features/residents/components/Residents';

export default function ResidentsPage({ query = '' }: { query?: string }) {
  return <Residents query={query} />;
}

