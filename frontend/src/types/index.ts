export type AlertStatus = 'UNREAD' | 'ACKNOWLEDGED' | 'RESOLVED';

export interface ApiAlert {
  id: string;
  incident_id: string | null;
  message: string;
  status: AlertStatus;
  created_at: string;
  resolved_at: string | null;
}

export interface ApiIncident {
  id: string;
  created_at: string;
  incident_metadata: {
    rule_triggered?: string;
    priority?: string;
    module?: string;
    camera_id?: string;
    timestamp?: string;
    detections?: any[];
  };
}

export interface ApiPerson {
  id: string;
  full_name: string;
  person_type: string;
  building?: string | null;
  apartment?: string | null;
  phone?: string | null;
  email?: string | null;
  valid_from?: string | null;
  valid_until?: string | null;
  created_at: string;
}

export interface ApiCamera {
  id: string;
  location: string;
  ip_address: string;
  status: string;
}

export interface EvidenceFile {
  object_name: string;
  size: number;
  last_modified?: string | null;
  content_type?: string | null;
}

export interface AlertCounts {
  unread: number;
  acknowledged: number;
  resolved: number;
}

