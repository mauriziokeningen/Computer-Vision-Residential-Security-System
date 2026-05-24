import React, { useState, useEffect } from 'react';
import { SidebarLayout, TabId } from './layouts/SidebarLayout';
import { useAlertWebSocket } from './hooks/useAlertWebSocket';

import DashboardPage from './pages/DashboardPage';
import LiveMonitoringPage from './pages/LiveMonitoringPage';
import AlertsPage from './pages/AlertsPage';
import IncidentsPage from './pages/IncidentsPage';
import ResidentsPage from './pages/ResidentsPage';
import AccessPage from './pages/AccessPage';
import SettingsPage from './pages/SettingsPage';

export default function App() {
  const [tab, setTab] = useState<TabId>(() => {
    const saved = window.localStorage.getItem('uh_security_active_tab');
    return (saved as TabId) || 'dashboard';
  });
  const [searchQuery, setSearchQuery] = useState('');

  const { alertCounts, isConnected, lastIncidentEvent } = useAlertWebSocket();

  useEffect(() => {
    window.localStorage.setItem('uh_security_active_tab', tab);
  }, [tab]);

  return (
    <SidebarLayout
      tab={tab}
      setTab={setTab}
      searchQuery={searchQuery}
      setSearchQuery={setSearchQuery}
      alertCounts={alertCounts}
      wsConnected={isConnected}
    >
      {tab === 'dashboard' && <DashboardPage alertCounts={alertCounts} />}
      {tab === 'live' && <LiveMonitoringPage />}
      {tab === 'alerts' && <AlertsPage query={searchQuery} />}
      {tab === 'incidents' && <IncidentsPage query={searchQuery} lastIncidentEvent={lastIncidentEvent} />}
      {tab === 'residents' && <ResidentsPage query={searchQuery} />}
      {tab === 'access' && <AccessPage onRegisterVisitor={() => setTab('residents')} />}
      {tab === 'settings' && <SettingsPage />}
    </SidebarLayout>
  );
}