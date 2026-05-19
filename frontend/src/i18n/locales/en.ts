import type { Dictionary } from './es';

export const en: Dictionary = {
  nav: {
    dashboard: 'Dashboard',
    live: 'Live Monitoring',
    alerts: 'Alerts',
    incidents: 'Incidents',
    residents: 'Residents',
    access: 'Access Control',
    settings: 'Settings',
    navigation: 'Navigation',
    sections: 'System Sections',
  },
  header: {
    title: 'UH Security · Panel',
    subtitle: 'Real-time monitoring · TT2',
    notifications: 'Notifications',
    searchPlaceholder: 'Search alerts, incidents or persons',
  },
  ws: {
    connected: 'Backend connected in real time',
    disconnected: 'No real-time backend connection available',
  },
  alerts: {
    unknownPersonDetected: 'Unknown person detected at {camera_id}',
  },
};