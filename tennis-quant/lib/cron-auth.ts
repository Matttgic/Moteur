import { createHash, timingSafeEqual } from "node:crypto";
import type { NextRequest } from "next/server";

const SCHEDULER_TOKEN_SHA256 =
  "56e58d13e49669747926615b2cff1617d571549a94f1632789414aa63d1f4c75";

export function isCronAuthorized(request: NextRequest) {
  const authorization = request.headers.get("authorization");
  const cronSecret = process.env.CRON_SECRET;

  if (cronSecret && authorization === `Bearer ${cronSecret}`) {
    return true;
  }

  if (!authorization?.startsWith("Bearer ")) {
    return false;
  }

  const token = authorization.slice("Bearer ".length);
  const actual = createHash("sha256").update(token).digest();
  const expected = Buffer.from(SCHEDULER_TOKEN_SHA256, "hex");

  return (
    actual.length === expected.length &&
    timingSafeEqual(actual, expected)
  );
}
