/**
 * @module SecurityDashboardRouter
 *
 * Root routing layer. Following the Feature-Sliced Design refactor,
 * App.tsx is reduced to its minimum responsibility: map the active tab
 * to a Page component, and wire the global alert WebSocket state into
 * the SidebarLayout.
 *
 * All API/network logic lives in src/api, all domain UI lives in
 * src/features, and the real-time WebSocket orchestration lives in
 * src/hooks/useAlertWebSocket.
 */

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

  // Custom hook manages WebSocket connection + global alert counts.
  const { alertCounts, isConnected } = useAlertWebSocket();

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
      {tab === 'incidents' && <IncidentsPage query={searchQuery} />}
      {tab === 'residents' && <ResidentsPage query={searchQuery} />}
      {tab === 'access' && <AccessPage onRegisterVisitor={() => setTab('residents')} />}
      {tab === 'settings' && <SettingsPage />}
    </SidebarLayout>
  );
}

