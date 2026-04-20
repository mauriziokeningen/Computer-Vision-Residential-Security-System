import React from 'react';
import { AccessGate } from '../features/access/components/AccessGate';

export default function AccessPage({ onRegisterVisitor }: { onRegisterVisitor: () => void }) {
  return <AccessGate onRegisterVisitor={onRegisterVisitor} />;
}

