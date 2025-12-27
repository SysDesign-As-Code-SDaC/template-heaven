/**
 * Run `build` or `dev` with `SKIP_ENV_VALIDATION` to skip env validation. This is especially
 * useful for Docker builds.
 */
await import("./src/env.js");

/** @type {import("next").NextConfig} */
const config = {
  reactStrictMode: true,

  /**
   * If you are using `appDir` then you must comment the below `i18n` config out.
   *
   * @see https://github.com/vercel/next.js/issues/41980
   */
  i18n: {
    locales: ["en"],
    defaultLocale: "en",
  },
  
  // Enable experimental features
  experimental: {
    // Enable server components logging
    serverComponentsExternalPackages: ["@prisma/client"],
  },
  
  // Optimize images
  images: {
    domains: ["localhost"],
    formats: ["image/webp", "image/avif"],
  },
  
  // Enable compression
  compress: true,
  
  // Enable SWC minification
  swcMinify: true,
  
  // Enable static optimization
  trailingSlash: false,
  
  // Enable source maps in production for better debugging
  productionBrowserSourceMaps: false,
};

export default config;
