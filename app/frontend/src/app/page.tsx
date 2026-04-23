import { API_BASE_URL } from '@/lib/config'
import HomeClient from '@/components/HomeClient'

// Revalidate data periodically (e.g., every 1 hour)
export const revalidate = 3600;

export default async function HomePage() {
  let stats = null;
  
  try {
    const res = await fetch(`${API_BASE_URL}/api/stats`, {
      next: { revalidate: 3600 }
    });
    if (res.ok) {
      stats = await res.json();
    }
  } catch (err) {
    console.error('Failed to fetch home page stats:', err);
  }

  return <HomeClient stats={stats} />
}
