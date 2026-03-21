import type {
  AnalyticsDetectionType,
  AnalyticsEventPerDay,
  AnalyticsHeatmap,
  AnalyticsSummary,
  AnalyticsTimeline,
  AnalyticsTopVisitor,
  AnalyticsVisitDuration
} from '$lib/types';
import { api } from './client';

export async function getEventsPerDay(days = 7): Promise<AnalyticsEventPerDay[]> {
  const res = await api.get('/analytics/events-per-day', { params: { days } });
  return res.data;
}

export async function getActivityHeatmap(days = 7): Promise<AnalyticsHeatmap[]> {
  const res = await api.get('/analytics/activity-heatmap', { params: { days } });
  return res.data;
}

export async function getTopVisitors(days = 30, limit = 10): Promise<AnalyticsTopVisitor[]> {
  const res = await api.get('/analytics/top-visitors', { params: { days, limit } });
  return res.data;
}

export async function getDetectionTypes(days = 7): Promise<AnalyticsDetectionType[]> {
  const res = await api.get('/analytics/detection-types', { params: { days } });
  return res.data;
}

export async function getVisitDurations(days = 30): Promise<AnalyticsVisitDuration[]> {
  const res = await api.get('/analytics/visit-durations', { params: { days } });
  return res.data;
}

export async function getAnalyticsSummary(days = 7): Promise<AnalyticsSummary> {
  const res = await api.get('/analytics/summary', { params: { days } });
  return res.data;
}

export async function getAnalyticsTimeline(days = 7, interval = 'hour'): Promise<AnalyticsTimeline[]> {
  const res = await api.get('/analytics/timeline', { params: { days, interval } });
  return res.data;
}
