import type { NextConfig } from "next";

const reactorInternalUrl = (
  process.env.REACTOR_INTERNAL_URL?.trim() || "http://localhost:8080"
).replace(/\/+$/, "");

const nextConfig: NextConfig = {
  async rewrites() {
    return [
      {
        source: "/reactor/:path*",
        destination: `${reactorInternalUrl}/:path*`,
      },
    ];
  },
};

export default nextConfig;
