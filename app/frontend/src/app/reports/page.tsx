import { API_BASE_URL } from '@/lib/config'
import ReportsClient from '@/components/ReportsClient'

export const revalidate = 3600;

export default async function ReportsPage() {
  let stats = null;
  let sectors: string[] = [];
  let statsError: string | null = null;
  
  try {
    const [statsRes, sectorsRes] = await Promise.all([
      fetch(`${API_BASE_URL}/api/stats`, { next: { revalidate: 3600 } }),
      fetch(`${API_BASE_URL}/api/sectors`, { next: { revalidate: 3600 } })
    ]);

    if (!statsRes.ok || !sectorsRes.ok) {
      throw new Error('Network response was not ok');
    }

    stats = await statsRes.json();
    sectors = await sectorsRes.json();
  } catch (err) {
    console.error('Failed to fetch reports page data:', err);
    statsError = 'Failed to load statistics. Please check the backend connection.';
  }

  return <ReportsClient stats={stats} sectors={sectors} statsError={statsError} />
}

