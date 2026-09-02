import { FastH3App } from "@/components/FastH3App";
import { readConfig } from "@/lib/config";

export const dynamic = "force-dynamic";

/** Resolve server-side connection settings and render the browser client. */
export default function Page() {
  return <FastH3App config={readConfig()} />;
}
