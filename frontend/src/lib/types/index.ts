export interface DetectionResponse {
  class_name: string;
  class_id: number;
  confidence: number;
  xcenter: number;
  ycenter: number;
  width: number;
  height: number;
}

export interface EventResponse {
  id: number;
  event_type: 'motion' | 'ding' | 'arrival' | 'departure';
  camera_name: string;
  occurred_at: string;
  snapshot_url: string | null;
  detection_summary: string | null;
  reference_name: string | null;
  display_name: string | null;
  caption: string | null;
  detections: DetectionResponse[];
  visit_id: number | null;
}

export interface EventListResponse {
  total: number;
  items: EventResponse[];
}

export interface VisitResponse {
  id: number;
  reference_name: string;
  display_name: string;
  camera_name: string;
  arrived_at: string;
  last_motion_at: string;
  departed_at: string | null;
  duration_minutes: number | null;
  snapshot_url: string | null;
  is_active: boolean;
}

export interface VisitListResponse {
  total: number;
  items: VisitResponse[];
}

export interface ReferenceResponse {
  uuid: string;
  name: string;
  display_name: string;
  category: string;
  visit_count: number;
  last_seen: string | null;
  sample_image_url: string | null;
}

export interface FaceProfileResponse {
  uuid: string;
  name: string;
  display_name: string;
  created_at: string;
  visit_count: number;
  last_seen: string | null;
  sample_image_url: string | null;
}

export interface ComponentStatus {
  status: 'ok' | 'warn' | 'fail' | 'missing' | 'off';
  detail: string;
}

export interface SystemStatus {
  database: ComponentStatus;
  gpu: ComponentStatus;
  yolo_model: ComponentStatus;
  ring_token: ComponentStatus;
  face_models: ComponentStatus;
  ollama: ComponentStatus;
  archive_dir: ComponentStatus;
  watcher_running: boolean;
  uptime_seconds: number;
}

export interface UnmatchedVehicle {
  cluster_id: string;
  sighting_count: number;
  first_seen: string;
  last_seen: string;
  camera_name: string;
  representative_crop_url: string | null;
  all_crop_urls: string[];
}

export interface UnmatchedFace {
  face_embedding_uuid: string;
  sighting_count: number;
  last_seen: string;
  camera_name: string;
  crop_url: string | null;
}

export interface AnalyticsEventPerDay { date: string; count: number; }
export interface AnalyticsHeatmap { hour: number; count: number; }
export interface AnalyticsTopVisitor { display_name: string; visit_count: number; last_seen: string; }
export interface AnalyticsDetectionType { class_name: string; count: number; percentage: number; }
export interface AnalyticsVisitDuration { bucket: string; count: number; }

export interface AnalyticsSummary {
  total_events: number;
  total_visits: number;
  active_visits: number;
  total_detections: number;
  cameras: string[];
  avg_visit_duration_minutes: number | null;
}

export interface AnalyticsTimeline {
  timestamp: string;
  count: number;
}
