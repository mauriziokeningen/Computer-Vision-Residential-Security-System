export const es = {
  nav: {
    dashboard: 'Dashboard',
    live: 'Monitoreo en vivo',
    alerts: 'Alertas',
    incidents: 'Incidentes',
    residents: 'Residentes',
    access: 'Control de Acceso',
    settings: 'Configuración',
    navigation: 'Navegación',
    sections: 'Secciones del sistema',
  },
  header: {
    title: 'Seguridad UH · Panel',
    subtitle: 'Monitoreo en tiempo real · TT2',
    notifications: 'Notificaciones',
    searchPlaceholder: 'Buscar alertas, incidentes o personas',
  },
  ws: {
    connected: 'Backend conectado en tiempo real',
    disconnected: 'Sin conexión al backend en tiempo real',
  },
  alerts: {
    unknownPersonDetected: 'Persona desconocida detectada en {camera_id}',
  },
} as const;

export type Dictionary = typeof es;