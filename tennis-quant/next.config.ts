import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  poweredByHeader: false,
  async rewrites() {
    return [{ source: "/court-vision", destination: "/court-vision/index.html" }];
  },
};

export default nextConfig;
