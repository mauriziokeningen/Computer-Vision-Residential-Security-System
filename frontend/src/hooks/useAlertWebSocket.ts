import { useState, useEffect } from 'react';
import { AlertCounts } from '../types';
import { buildWsUrl, apiFetch } from '../api/client';

export function useAlertWebSocket() {
  const [alertCounts, setAlertCounts] = useState<AlertCounts>({ unread: 0, acknowledged: 0, resolved: 0 });
  const [isConnected, setIsConnected] = useState(false);
  const [lastIncidentEvent, setLastIncidentEvent] = useState<number>(0);

  // 1. Initial Data Fetch
  useEffect(() => {
    Promise.all([
      apiFetch<{ count: number }>('/alerts/count?status=UNREAD'),
      apiFetch<{ count: number }>('/alerts/count?status=ACKNOWLEDGED'),
      apiFetch<{ count: number }>('/alerts/count?status=RESOLVED'),
    ])
      .then(([u, a, r]) => setAlertCounts({ unread: u.count, acknowledged: a.count, resolved: r.count }))
      .catch(() => {});
  }, []);

  // 2. Real-Time WebSocket Connection
  useEffect(() => {
    let ws: WebSocket;
    let reconnectTimeout: ReturnType<typeof setTimeout>;
    let isMounted = true;

    const connect = () => {
      if (!isMounted) return;
      ws = new WebSocket(buildWsUrl('/ws/alerts'));

      ws.onopen = () => { if (isMounted) setIsConnected(true); };

      ws.onmessage = (ev) => {
        try {
          const msg = JSON.parse(ev.data);
          if (msg.event_type === 'ALERT_COUNT_UPDATE') {
            setAlertCounts({
              unread: msg.data.unread,
              acknowledged: msg.data.acknowledged,
              resolved: msg.data.resolved,
            });
          } else if (msg.event_type === 'NEW_ALERT' || msg.event_type === 'ALERT_STATUS_CHANGED') {
            setLastIncidentEvent(Date.now()); // 👈 señal global para todos los componentes
            Promise.all([
              apiFetch<{ count: number }>('/alerts/count?status=UNREAD'),
              apiFetch<{ count: number }>('/alerts/count?status=ACKNOWLEDGED'),
              apiFetch<{ count: number }>('/alerts/count?status=RESOLVED'),
            ]).then(([u, a, r]) => setAlertCounts({ unread: u.count, acknowledged: a.count, resolved: r.count })).catch(() => {});
          }
        } catch {}
      };

      ws.onclose = () => {
        if (isMounted) {
          setIsConnected(false);
          reconnectTimeout = setTimeout(connect, 3000);
        }
      };

      ws.onerror = () => ws.close();
    };

    connect();
    return () => {
      isMounted = false;
      clearTimeout(reconnectTimeout);
      if (ws) ws.close();
    };
  }, []);

  return { alertCounts, isConnected, lastIncidentEvent };
}