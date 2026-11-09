import ratingsData from '../data/cikm-2026-ratings.json';
import scheduleData from '../data/cikm-2026-schedule.json';
import rubricData from '../data/cikm-2026-rating-rubric.json';

export const rubric = rubricData;

export const grades = {
  strong: 'Крутые', solid: 'Нормальные', weaker: 'Не очень сильные'
} as const;
export type Grade = keyof typeof grades;
export type Rating = {
  selectionId: string; interest: number; evidence: number; importance: number;
  group: Grade; reason: string;
};
export type Appearance = {
  kind: 'oral' | 'poster'; date: string; dayLabel: string; start: string; end: string;
  timePrecision: 'session' | 'talk'; room: string | null; venue?: string;
  sessionCode: string; sessionTitle: string; track: string; sourceUrl: string;
  board: string | null;
};
export type PaperSchedule = {
  selectionId: string;
  presentation: 'talk_and_poster' | 'poster_only' | 'oral_only' | 'unknown';
  appearances: Appearance[];
};
export const schedule = scheduleData as {
  sourceUrl: string; checkedAt: string; timezone: string; notes: string[]; papers: PaperSchedule[];
};
export const ratings = ratingsData as Rating[];
export const ratingsById = new Map(ratings.map(r => [r.selectionId, r]));
export const scheduleById = new Map(schedule.papers.map(p => [p.selectionId, p]));
export const days = [
  { date: '2026-11-09', label: 'Понедельник, 9 ноября', short: 'Пн, 9 ноября' },
  { date: '2026-11-10', label: 'Вторник, 10 ноября', short: 'Вт, 10 ноября' },
  { date: '2026-11-11', label: 'Среда, 11 ноября', short: 'Ср, 11 ноября' },
];
export const formats = {
  poster_only: 'Только постер', oral_only: 'Доклад · постер не указан',
  talk_and_poster: 'Доклад и постер', unknown: 'Формат не уточнён',
};
export function dayLabel(date: string) {
  return days.find(day => day.date === date)?.short ?? date;
}
