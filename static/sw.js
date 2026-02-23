const CACHE_NAME = 'fashion-tryon-v3';
const ASSETS_TO_CACHE = [
  '/static/manifest.json',
  '/static/icons/icon-192.png',
  '/static/icons/icon-512.png'
];

// Install: cache only static assets, not HTML
self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      return cache.addAll(ASSETS_TO_CACHE);
    })
  );
  self.skipWaiting();
});

// Activate: clean old caches immediately
self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) => {
      return Promise.all(
        keys.filter((key) => key !== CACHE_NAME).map((key) => caches.delete(key))
      );
    })
  );
  self.clients.claim();
});

// Fetch: ALWAYS get HTML from network, only cache static assets
self.addEventListener('fetch', (event) => {
  if (event.request.method !== 'GET') return;

  // Never cache API calls or HTML pages
  const url = new URL(event.request.url);
  if (event.request.url.includes('/api/') || 
      event.request.mode === 'navigate' ||
      url.pathname === '/' ||
      url.pathname === '/admin') {
    event.respondWith(fetch(event.request));
    return;
  }

  // Static assets: cache first, network fallback
  event.respondWith(
    caches.match(event.request).then((cached) => {
      return cached || fetch(event.request).then((response) => {
        if (response.status === 200) {
          const clone = response.clone();
          caches.open(CACHE_NAME).then((cache) => cache.put(event.request, clone));
        }
        return response;
      });
    })
  );
});