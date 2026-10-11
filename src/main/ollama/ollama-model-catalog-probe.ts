import { readFile } from 'node:fs/promises'
import { join } from 'node:path'
import { parse as parseYaml } from 'yaml'
import type { AgentSessionAccountHome } from '../../shared/agent-session-account-home'
import { isLegacyAgentSessionAccountHome } from '../../shared/agent-session-account-home'
import type { AgentSessionModelOption } from '../../shared/agent-session-wire'
import type {
  AgentModelCatalogProbe,
  AgentModelCatalogSuccess
} from '../native-chat/agent-model-catalog/agent-model-catalog-store'
import { findEmbeddingOnlyOllamaModels } from './ollama-model-capabilities'

const DEFAULT_OLLAMA_URL = 'http://localhost:11434'
const FETCH_TIMEOUT_MS = 2000

type GooseConfig = {
  GOOSE_PROVIDER?: unknown
  GOOSE_MODEL?: unknown
  OLLAMA_HOST?: unknown
  active_provider?: unknown
  providers?: unknown
  allowed_models?: unknown
}

type OllamaTagsResponse = {
  models?:
    | {
        name: string
        size?: number
        details?: {
          family?: string
          parameter_size?: string
          quantization_level?: string
        } | null
      }[]
    | null
}

function record(value: unknown): Record<string, unknown> | null {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    return null
  }
  return Object.fromEntries(Object.entries(value))
}

function nonEmptyString(value: unknown): string | null {
  return typeof value === 'string' && value.trim() ? value.trim() : null
}

async function readGooseConfig(accountHome: AgentSessionAccountHome): Promise<GooseConfig> {
  if (!isLegacyAgentSessionAccountHome(accountHome)) {
    return {}
  }
  try {
    const root = record(parseYaml(await readFile(join(accountHome.path, 'config.yaml'), 'utf8')))
    return root
      ? {
          GOOSE_PROVIDER: root.GOOSE_PROVIDER,
          GOOSE_MODEL: root.GOOSE_MODEL,
          OLLAMA_HOST: root.OLLAMA_HOST,
          active_provider: root.active_provider,
          providers: root.providers,
          allowed_models: root.allowed_models
        }
      : {}
  } catch {
    return {}
  }
}

function activeProviderConfig(
  config: GooseConfig,
  activeProvider: string | null
): Record<string, unknown> | null {
  const providers = record(config.providers)
  if (!activeProvider || !providers) {
    return null
  }
  const matchingName = Object.keys(providers).find(
    (name) => name.toLowerCase() === activeProvider.toLowerCase()
  )
  return matchingName ? record(providers[matchingName]) : null
}

function hasExplicitPort(raw: string): boolean {
  const authority = raw.replace(/^[a-z][a-z\d+.-]*:\/\//i, '').split(/[/?#]/, 1)[0] ?? ''
  const host = authority.split('@').at(-1) ?? authority
  return host.startsWith('[')
    ? host.slice(host.indexOf(']') + 1).startsWith(':')
    : host.includes(':')
}

function ollamaUrl(config: GooseConfig, launchEnv: NodeJS.ProcessEnv, providerName: string): URL {
  const provider = activeProviderConfig(config, providerName)
  const configuredHost =
    nonEmptyString(launchEnv.OLLAMA_HOST) ??
    nonEmptyString(config.OLLAMA_HOST) ??
    nonEmptyString(provider?.host) ??
    nonEmptyString(provider?.base_url)
  const raw = configuredHost ?? DEFAULT_OLLAMA_URL
  const url = new URL(/^[a-z][a-z\d+.-]*:\/\//i.test(raw) ? raw : `http://${raw}`)
  if (url.protocol !== 'http:' && url.protocol !== 'https:') {
    throw new Error('Ollama endpoint must use HTTP or HTTPS')
  }
  // URL defaults HTTP to port 80; Ollama's service port is 11434.
  if (url.protocol === 'http:' && !url.port && !hasExplicitPort(raw)) {
    url.port = '11434'
  }
  url.pathname = `${url.pathname.replace(/\/$/, '')}/api/tags`
  url.search = ''
  url.hash = ''
  return url
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) {
    return `${bytes} B`
  }
  if (bytes < 1024 ** 2) {
    return `${(bytes / 1024).toFixed(1)} KB`
  }
  if (bytes < 1024 ** 3) {
    return `${(bytes / 1024 ** 2).toFixed(1)} MB`
  }
  return `${(bytes / 1024 ** 3).toFixed(1)} GB`
}

function modelLabel(name: string): string {
  const [base, tag] = name.split(':', 2)
  const parts = (base ?? name).split('/')
  const model = parts.pop() ?? name
  const library = parts.pop()
  const display = library ? `${model} (${library})` : model
  return tag && tag !== 'latest' ? `${display} (${tag})` : display
}

function allowedModelNames(config: GooseConfig, providerName: string): Set<string> | null {
  const providerModels = activeProviderConfig(config, providerName)?.models
  const configured = Array.isArray(config.allowed_models)
    ? config.allowed_models
    : Array.isArray(providerModels)
      ? providerModels
      : null
  if (!configured) {
    return null
  }
  return new Set(
    configured
      .map((entry) => nonEmptyString(entry) ?? nonEmptyString(record(entry)?.name))
      .filter((name): name is string => name !== null)
  )
}

function configuredModel(
  config: GooseConfig,
  launchEnv: NodeJS.ProcessEnv,
  providerName: string
): string | null {
  return (
    nonEmptyString(launchEnv.GOOSE_MODEL) ??
    nonEmptyString(activeProviderConfig(config, providerName)?.model) ??
    nonEmptyString(activeProviderConfig(config, providerName)?.default_model) ??
    nonEmptyString(config.GOOSE_MODEL)
  )
}

function modelOptions(
  response: OllamaTagsResponse,
  excludedModelNames: ReadonlySet<string>,
  config: GooseConfig,
  launchEnv: NodeJS.ProcessEnv,
  providerName: string
): AgentSessionModelOption[] {
  const allowed = allowedModelNames(config, providerName)
  const defaultModel = configuredModel(config, launchEnv, providerName)
  return (response.models ?? [])
    .filter((model) => !excludedModelNames.has(model.name))
    .filter((model) => !allowed || allowed.has(model.name))
    .map((model) => {
      const details = model.details
      const description = [
        'Ollama',
        details?.family,
        details?.parameter_size,
        details?.quantization_level,
        model.size === undefined ? undefined : formatBytes(model.size)
      ]
        .filter((part): part is string => Boolean(part))
        .join(' · ')
      return {
        // Goose accepts the Ollama name as-is through its ACP model selector.
        id: model.name,
        label: modelLabel(model.name),
        description,
        isDefault: model.name === defaultModel,
        efforts: []
      }
    })
}

function parseTagsResponse(value: unknown): OllamaTagsResponse {
  const root = record(value)
  const models = root?.models
  if (!Array.isArray(models)) {
    return {}
  }
  return {
    models: models.flatMap((value) => {
      const model = record(value)
      const name = nonEmptyString(model?.name)
      if (!model || !name) {
        return []
      }
      const details = record(model.details)
      const size = typeof model.size === 'number' ? model.size : undefined
      const family = nonEmptyString(details?.family)
      const parameterSize = nonEmptyString(details?.parameter_size)
      const quantizationLevel = nonEmptyString(details?.quantization_level)
      return [
        {
          name,
          ...(size !== undefined ? { size } : {}),
          ...(details
            ? {
                details: {
                  ...(family ? { family } : {}),
                  ...(parameterSize ? { parameter_size: parameterSize } : {}),
                  ...(quantizationLevel ? { quantization_level: quantizationLevel } : {})
                }
              }
            : {})
        }
      ]
    })
  }
}

export type OllamaModelCatalogProbeDeps = {
  resolveLaunchEnv?: () => NodeJS.ProcessEnv | Promise<NodeJS.ProcessEnv>
  fetch?: typeof fetch
  timeoutMs?: number
}

/** Lists local Ollama models for the Goose ACP catalog; an unavailable endpoint is an optional probe failure. */
export function createOllamaModelCatalogProbe(
  deps: OllamaModelCatalogProbeDeps = {}
): AgentModelCatalogProbe {
  return async (accountHome, options): Promise<AgentModelCatalogSuccess> => {
    const config = await readGooseConfig(accountHome)
    const launchEnv = (await deps.resolveLaunchEnv?.()) ?? process.env
    const providerName =
      nonEmptyString(launchEnv.GOOSE_PROVIDER) ??
      nonEmptyString(config.GOOSE_PROVIDER) ??
      nonEmptyString(config.active_provider)
    if (providerName?.toLowerCase() !== 'ollama') {
      throw new Error('Goose is not configured to use Ollama')
    }

    const controller = new AbortController()
    const abort = (): void => controller.abort()
    if (options?.signal?.aborted) {
      abort()
    }
    options?.signal?.addEventListener('abort', abort, { once: true })
    const timer = setTimeout(() => controller.abort(), deps.timeoutMs ?? FETCH_TIMEOUT_MS)
    try {
      const fetcher = deps.fetch ?? fetch
      const tagsUrl = ollamaUrl(config, launchEnv, providerName)
      const response = await fetcher(tagsUrl, {
        signal: controller.signal,
        headers: { Accept: 'application/json' }
      })
      if (!response.ok) {
        throw new Error(`Ollama API responded with status ${response.status}`)
      }
      const tags = parseTagsResponse(await response.json())
      const embeddingOnly = await findEmbeddingOnlyOllamaModels(
        tags.models,
        tagsUrl,
        fetcher,
        controller.signal
      )
      controller.signal.throwIfAborted()
      const listed = modelOptions(tags, embeddingOnly, config, launchEnv, providerName)
      if (listed.length === 0) {
        throw new Error('Ollama listed no models')
      }
      return { models: listed, origin: 'probe' }
    } finally {
      clearTimeout(timer)
      options?.signal?.removeEventListener('abort', abort)
    }
  }
}
