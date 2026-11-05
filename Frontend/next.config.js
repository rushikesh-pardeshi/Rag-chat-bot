/**
 * frontend/next.config.js
 * Role: Next.js build/runtime config.
 *
 * Security headers are set here rather than in a proxy so they hold in local development
 * too. The API lives on a different origin (:8000) and is called directly with
 * credentials, so it is listed in connect-src rather than proxied.
 *
 * ## Two directives differ between dev and production, and only two
 *
 * React's **development** build uses `eval` to rebuild component stacks across the
 * server/client boundary, so a policy without `'unsafe-eval'` fills the console with
 * *"eval is not supported in this environment"* and loses the stack traces that make an error
 * traceable. React never uses `eval` in production.
 *
 * (Written without the call parentheses on purpose: `test_xss.py::TestTheFrontendDoesNotOptOut`
 * greps every frontend source file for a *call* to `eval` and has no comment exemption, which
 * is the right default — a scan that skipped comments could be silenced by one.)
 *
 * So `'unsafe-eval'` is added when `NODE_ENV` is not `production`, and in no other case. The
 * production policy is byte-for-byte what it was — which matters, because `'unsafe-eval'` is a
 * materially weaker posture than `'unsafe-inline'`: it lets an injected string become code. A
 * developer machine already runs the dev server, the source and the toolchain; a production
 * browser does not, and does not get the exemption.
 *
 * The second is `connect-src ws:`, for the hot-reload WebSocket. See `devConnectSrc` below.
 *
 * `backend/tests/security/test_xss.py::TestTheContentSecurityPolicy` pins that split — it reads
 * this file and fails if `'unsafe-eval'` or a `ws:` allowance ever appears on a line that is not
 * gated on `isProduction`. Keep each flag on the same line as its gate for that reason.
 */

/** @type {import('next').NextConfig} */
const apiBase = process.env.NEXT_PUBLIC_API_BASE_URL ?? 'http://localhost:8000';

/** `next build` sets this to `production`; `next dev` leaves it `development`. */
const isProduction = process.env.NODE_ENV === 'production';

// Development only, and on one line so the gate is visible next to what it allows.
const devScriptSrc = isProduction ? '' : " 'unsafe-eval'";

// Development only, same reasoning and same shape. `next dev` hot-reloads over a WebSocket to
// `ws://<host>/_next/hmr`, and WebSocket URLs are matched by `connect-src` — not `default-src`,
// and there is no separate directive for them.
//
// CSP Level 3 does say `'self'` should match a `ws:` URL on the same host and port as an
// `http:` document, but that was a late addition to the spec and is not something to rely on
// across browsers: the symptom when it is not honoured is a console filling with
// `WebSocket connection to 'ws://localhost:3000/_next/hmr' failed` and hot reload silently
// not working. Naming the scheme costs nothing here and removes the doubt.
//
// Both schemes, because the page's scheme decides which one the HMR client uses: `ws:` for a
// plain-HTTP dev server, `wss:` once anything terminates TLS in front of it. Listing only
// `ws:` left `WEB_MODE=dev` behind the POC's HTTPS front door still blocked, which is the
// configuration this was first reported from.
//
// Never in production: `next start` serves a built app and opens no HMR socket, so there is
// nothing for this to enable there and it would only widen where the page may talk to.
const devConnectSrc = isProduction ? '' : ' ws: wss:';

const securityHeaders = [
  { key: 'X-Content-Type-Options', value: 'nosniff' },
  { key: 'X-Frame-Options', value: 'DENY' },
  { key: 'Referrer-Policy', value: 'strict-origin-when-cross-origin' },
  { key: 'Permissions-Policy', value: 'camera=(), microphone=(), geolocation=()' },
  {
    key: 'Content-Security-Policy',
    value: [
      "default-src 'self'",
      // Next injects inline bootstrap scripts; style-src inline is Tailwind's runtime.
      `script-src 'self' 'unsafe-inline'${devScriptSrc}`,
      "style-src 'self' 'unsafe-inline'",
      "img-src 'self' data: blob:",
      "font-src 'self' data:",
      `connect-src 'self' ${apiBase}${devConnectSrc}`,
      // Presigned S3 document downloads open in a new tab, never framed.
      "frame-ancestors 'none'",
      "object-src 'none'",
      "base-uri 'self'",
      "form-action 'self'",
    ].join('; '),
  },
];

const nextConfig = {
  reactStrictMode: true,
  poweredByHeader: false,
  async headers() {
    return [{ source: '/:path*', headers: securityHeaders }];
  },
};

module.exports = nextConfig;
