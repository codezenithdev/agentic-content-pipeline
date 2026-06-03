/** @type {import('next').NextConfig} */
const nextConfig = {
  // Lint is run separately; don't let lint rules block the production build.
  // TypeScript type-checking still runs and will fail the build on real type errors.
  eslint: { ignoreDuringBuilds: true },
};

export default nextConfig;
