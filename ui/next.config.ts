import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // dev access through the local Caddy proxy (see scripts/Caddyfile)
  allowedDevOrigins: ["researcher.localhost"],
};

export default nextConfig;
