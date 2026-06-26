import { API_BASE_URL } from '@/lib/config'
import HomeClient from '@/components/HomeClient'

export const dynamic = 'force-dynamic'
export const revalidate = 0

export default async function HomePage() {
  let stats = null;
  
  try {
    const res = await fetch(`${API_BASE_URL}/api/stats`, { cache: 'no-store' });
    if (res.ok) {
      stats = await res.json();
    }
  } catch (err) {
    console.error('Failed to fetch home page stats:', err);
  }

  return <HomeClient stats={stats} />
}
