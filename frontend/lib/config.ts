export type Mode = "local" | "hosted";

export interface FrontendConfig {
  mode: Mode;
  modelName: string;
  apiUrl?: string;
  defaultSessionId?: string;
}

const DEFAULT_MODEL_NAME = "fast-h3";
const DEFAULT_LOCAL_URL = "/reactor";

/** Resolve connection settings without exposing an API key to the browser. */
export function readConfig(): FrontendConfig {
  const modelName =
    process.env.REACTOR_MODEL_NAME?.trim() || DEFAULT_MODEL_NAME;
  const defaultSessionId = process.env.REACTOR_SESSION_ID?.trim() || undefined;

  if (process.env.REACTOR_API_KEY?.trim()) {
    return {
      mode: "hosted",
      modelName,
      apiUrl: process.env.REACTOR_API_URL?.trim() || undefined,
      defaultSessionId,
    };
  }

  return {
    mode: "local",
    modelName,
    apiUrl: process.env.REACTOR_LOCAL_URL?.trim() || DEFAULT_LOCAL_URL,
    defaultSessionId,
  };
}
