import fs from "node:fs";
import path from "node:path";
import { listAgentIds, resolveAgentDir } from "../agents/agent-scope.js";
import type { OpenClawConfig } from "../config/config.js";
import { pathExists, resolveUserPath } from "../utils.js";
import {
  listAuthProfileStorePaths as listAuthProfileStorePathsFromAuthStorePaths,
  listAuthProfileStorePathsAsync as listAuthProfileStorePathsFromAuthStorePathsAsync,
} from "./auth-store-paths.js";
import { parseEnvValue } from "./shared.js";

export function parseEnvAssignmentValue(raw: string): string {
  return parseEnvValue(raw);
}

export function listAuthProfileStorePaths(config: OpenClawConfig, stateDir: string): string[] {
  return listAuthProfileStorePathsFromAuthStorePaths(config, stateDir);
}

export async function listAuthProfileStorePathsAsync(
  config: OpenClawConfig,
  stateDir: string,
): Promise<string[]> {
  return listAuthProfileStorePathsFromAuthStorePathsAsync(config, stateDir);
}

export function listLegacyAuthJsonPaths(stateDir: string): string[] {
  const out: string[] = [];
  const agentsRoot = path.join(resolveUserPath(stateDir), "agents");
  if (!fs.existsSync(agentsRoot)) {
    return out;
  }
  for (const entry of fs.readdirSync(agentsRoot, { withFileTypes: true })) {
    if (!entry.isDirectory()) {
      continue;
    }
    const candidate = path.join(agentsRoot, entry.name, "agent", "auth.json");
    if (fs.existsSync(candidate)) {
      out.push(candidate);
    }
  }
  return out;
}

export async function listLegacyAuthJsonPathsAsync(stateDir: string): Promise<string[]> {
  const out: string[] = [];
  const agentsRoot = path.join(resolveUserPath(stateDir), "agents");
  if (!(await pathExists(agentsRoot))) {
    return out;
  }
  const entries = await fs.promises.readdir(agentsRoot, { withFileTypes: true });
  for (const entry of entries) {
    if (!entry.isDirectory()) {
      continue;
    }
    const candidate = path.join(agentsRoot, entry.name, "agent", "auth.json");
    if (await pathExists(candidate)) {
      out.push(candidate);
    }
  }
  return out;
}

export function listAgentModelsJsonPaths(config: OpenClawConfig, stateDir: string): string[] {
  const paths = new Set<string>();
  paths.add(path.join(resolveUserPath(stateDir), "agents", "main", "agent", "models.json"));

  const agentsRoot = path.join(resolveUserPath(stateDir), "agents");
  if (fs.existsSync(agentsRoot)) {
    for (const entry of fs.readdirSync(agentsRoot, { withFileTypes: true })) {
      if (!entry.isDirectory()) {
        continue;
      }
      paths.add(path.join(agentsRoot, entry.name, "agent", "models.json"));
    }
  }

  for (const agentId of listAgentIds(config)) {
    if (agentId === "main") {
      paths.add(path.join(resolveUserPath(stateDir), "agents", "main", "agent", "models.json"));
      continue;
    }
    const agentDir = resolveAgentDir(config, agentId);
    paths.add(path.join(resolveUserPath(agentDir), "models.json"));
  }

  return [...paths];
}

export async function listAgentModelsJsonPathsAsync(
  config: OpenClawConfig,
  stateDir: string,
): Promise<string[]> {
  const paths = new Set<string>();
  paths.add(path.join(resolveUserPath(stateDir), "agents", "main", "agent", "models.json"));

  const agentsRoot = path.join(resolveUserPath(stateDir), "agents");
  if (await pathExists(agentsRoot)) {
    const entries = await fs.promises.readdir(agentsRoot, { withFileTypes: true });
    for (const entry of entries) {
      if (!entry.isDirectory()) {
        continue;
      }
      paths.add(path.join(agentsRoot, entry.name, "agent", "models.json"));
    }
  }

  for (const agentId of listAgentIds(config)) {
    if (agentId === "main") {
      paths.add(path.join(resolveUserPath(stateDir), "agents", "main", "agent", "models.json"));
      continue;
    }
    const agentDir = resolveAgentDir(config, agentId);
    paths.add(path.join(resolveUserPath(agentDir), "models.json"));
  }

  return [...paths];
}

export function readJsonObjectIfExists(filePath: string): {
  value: Record<string, unknown> | null;
  error?: string;
} {
  if (!fs.existsSync(filePath)) {
    return { value: null };
  }
  try {
    const raw = fs.readFileSync(filePath, "utf8");
    const parsed = JSON.parse(raw) as unknown;
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
      return { value: null };
    }
    return { value: parsed as Record<string, unknown> };
  } catch (err) {
    return {
      value: null,
      error: err instanceof Error ? err.message : String(err),
    };
  }
}

export async function readJsonObjectIfExistsAsync(filePath: string): Promise<{
  value: Record<string, unknown> | null;
  error?: string;
}> {
  if (!(await pathExists(filePath))) {
    return { value: null };
  }
  try {
    const raw = await fs.promises.readFile(filePath, "utf8");
    const parsed = JSON.parse(raw) as unknown;
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
      return { value: null };
    }
    return { value: parsed as Record<string, unknown> };
  } catch (err) {
    return {
      value: null,
      error: err instanceof Error ? err.message : String(err),
    };
  }
}
