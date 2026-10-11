import type { NextConfig } from "next";

const backendOrigin = process.env.TENNIS_BACKEND_ORIGIN?.replace(/\/$/, "");

const nextConfig: NextConfig = {
  poweredByHeader: false,
  async rewrites() {
    return {
      beforeFiles: backendOrigin
        ? [
            {
              source: "/api/:path*",
              destination: `${backendOrigin}/api/:path*`,
            },
          ]
        : [],
      afterFiles: [
        { source: "/court-vision", destination: "/court-vision/index.html" },
      ],
      fallback: [],
    };
  },
};

export default nextConfig;
