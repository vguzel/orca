import { mkdtemp, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, describe, expect, it, vi } from 'vitest'
import type { AgentSessionAccountHome } from '../../shared/agent-session-account-home'
import { createOllamaModelCatalogProbe } from './ollama-model-catalog-probe'

const temporaryDirectories: string[] = []

async function gooseHome(config: string): Promise<AgentSessionAccountHome> {
  const path = await mkdtemp(join(tmpdir(), 'orca-goose-'))
  temporaryDirectories.push(path)
  await writeFile(join(path, 'config.yaml'), config, 'utf8')
  return { variable: 'GOOSE_CONFIG_DIR', path }
}

function tagsResponse(names: readonly string[]): Response {
  return new Response(
    JSON.stringify({ models: names.map((name) => ({ name, size: 1024, details: null })) }),
    { status: 200, headers: { 'Content-Type': 'application/json' } }
  )
}

afterEach(async () => {
  await Promise.all(temporaryDirectories.splice(0).map((path) => rm(path, { recursive: true })))
  vi.restoreAllMocks()
})

describe('Goose Ollama model catalog probe', () => {
  it('uses the configured host, filters allowed models, and marks the Goose default', async () => {
    const home = await gooseHome(`
active_provider: ollama
OLLAMA_HOST: http://127.0.0.1:11500
providers:
  ollama:
    model: llama3.2:latest
allowed_models:
  - llama3.2:latest
`)
    const fetch = vi
      .fn<typeof globalThis.fetch>()
      .mockResolvedValue(tagsResponse(['llama3.2:latest', 'qwen3:latest']))
    const probe = createOllamaModelCatalogProbe({ fetch })

    const result = await probe(home)

    expect(fetch).toHaveBeenCalledWith(
      new URL('http://127.0.0.1:11500/api/tags'),
      expect.objectContaining({ headers: { Accept: 'application/json' } })
    )
    expect(result.models).toEqual([
      expect.objectContaining({
        id: 'llama3.2:latest',
        label: 'llama3.2',
        isDefault: true,
        description: expect.stringContaining('Ollama')
      })
    ])
  })

  it('uses Ollama’s default port when Goose config omits one', async () => {
    const home = await gooseHome(`
active_provider: ollama
OLLAMA_HOST: http://localhost
`)
    const fetch = vi
      .fn<typeof globalThis.fetch>()
      .mockResolvedValue(tagsResponse(['llama3.2:latest']))
    const probe = createOllamaModelCatalogProbe({ fetch })

    await probe(home)

    expect(fetch).toHaveBeenCalledWith(
      new URL('http://localhost:11434/api/tags'),
      expect.objectContaining({ headers: { Accept: 'application/json' } })
    )
  })

  it('omits models that Ollama identifies as embedding-only', async () => {
    const home = await gooseHome('GOOSE_PROVIDER: ollama')
    const fetch = vi.fn<typeof globalThis.fetch>(async (input, init) => {
      const url = new URL(String(input))
      if (url.pathname.endsWith('/api/tags')) {
        return tagsResponse(['bge-m3:latest', 'llama3.2:latest', 'uncertain-model:latest'])
      }
      const request: unknown = JSON.parse(String(init?.body))
      const modelName =
        typeof request === 'object' &&
        request !== null &&
        'model' in request &&
        typeof request.model === 'string'
          ? request.model
          : ''
      const capabilities =
        modelName === 'bge-m3:latest'
          ? ['embedding']
          : modelName === 'llama3.2:latest'
            ? ['completion', 'tools']
            : undefined
      return new Response(JSON.stringify(capabilities ? { capabilities } : {}), {
        status: 200,
        headers: { 'Content-Type': 'application/json' }
      })
    })
    const probe = createOllamaModelCatalogProbe({ fetch })

    const result = await probe(home)

    expect(result.models.map((model) => model.id)).toEqual([
      'llama3.2:latest',
      'uncertain-model:latest'
    ])
    expect(fetch).toHaveBeenCalledWith(
      new URL('http://localhost:11434/api/show'),
      expect.objectContaining({ method: 'POST' })
    )
  })

  it('returns a large installed model catalog', async () => {
    const home = await gooseHome('GOOSE_PROVIDER: ollama')
    const names = Array.from({ length: 128 }, (_, index) => `model-${index}:latest`)
    const probe = createOllamaModelCatalogProbe({
      fetch: vi.fn<typeof globalThis.fetch>().mockResolvedValue(tagsResponse(names))
    })

    const result = await probe(home)

    expect(result.models).toHaveLength(128)
  })

  it('treats an empty Ollama catalog as a failed optional probe', async () => {
    const home = await gooseHome('GOOSE_PROVIDER: ollama')
    const probe = createOllamaModelCatalogProbe({
      fetch: vi.fn<typeof globalThis.fetch>().mockResolvedValue(tagsResponse([]))
    })

    await expect(probe(home)).rejects.toThrow('Ollama listed no models')
  })

  it('surfaces HTTP failures to the catalog failure cache', async () => {
    const home = await gooseHome('GOOSE_PROVIDER: ollama')
    const probe = createOllamaModelCatalogProbe({
      fetch: vi.fn<typeof globalThis.fetch>().mockResolvedValue(new Response(null, { status: 503 }))
    })

    await expect(probe(home)).rejects.toThrow('Ollama API responded with status 503')
  })

  it('stops the request when the catalog service aborts', async () => {
    const home = await gooseHome('GOOSE_PROVIDER: ollama')
    const fetch = vi.fn<typeof globalThis.fetch>(async (_input, init) => {
      return new Promise<Response>((_resolve, reject) => {
        init?.signal?.addEventListener('abort', () => reject(new Error('aborted')), { once: true })
      })
    })
    const controller = new AbortController()
    const probe = createOllamaModelCatalogProbe({ fetch })
    const pending = probe(home, { signal: controller.signal })
    await vi.waitFor(() => expect(fetch).toHaveBeenCalled())
    controller.abort()

    await expect(pending).rejects.toThrow('aborted')
  })

  it('does not return a partial catalog when aborting model capability discovery', async () => {
    const home = await gooseHome('GOOSE_PROVIDER: ollama')
    const fetch = vi.fn<typeof globalThis.fetch>(async (input, init) => {
      if (new URL(String(input)).pathname.endsWith('/api/tags')) {
        return tagsResponse(['llama3.2:latest'])
      }
      return new Promise<Response>((_resolve, reject) => {
        init?.signal?.addEventListener('abort', () => reject(new Error('aborted')), { once: true })
      })
    })
    const controller = new AbortController()
    const probe = createOllamaModelCatalogProbe({ fetch })
    const pending = probe(home, { signal: controller.signal })
    await vi.waitFor(() => expect(fetch).toHaveBeenCalledTimes(2))
    controller.abort()

    await expect(pending).rejects.toThrow('aborted')
  })
})
