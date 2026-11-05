/**
 * frontend/postcss.config.mjs
 * Role: PostCSS pipeline. Tailwind v4 is a single plugin; there is no tailwind.config.js —
 * theme and tokens are declared in CSS via @theme in app/globals.css.
 */
const config = {
  plugins: {
    '@tailwindcss/postcss': {},
  },
};

export default config;
