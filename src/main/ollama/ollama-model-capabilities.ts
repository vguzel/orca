type OllamaModelReference = { name: string }

function record(value: unknown): Record<string, unknown> | null {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    return null
  }
  return Object.fromEntries(Object.entries(value))
}

function showUrl(tagsUrl: URL): URL {
  const url = new URL(tagsUrl)
  url.pathname = `${url.pathname.replace(/\/api\/tags$/, '')}/api/show`
  return url
}

/** Keeps model selection chat-focused while retaining models when Ollama cannot classify them. */
export async function findEmbeddingOnlyOllamaModels(
  models: OllamaModelReference[] | null | undefined,
  tagsUrl: URL,
  fetcher: typeof fetch,
  signal: AbortSignal
): Promise<Set<string>> {
  const candidates = models ?? []
  const excluded = new Set<string>()
  let nextIndex = 0
  const workers = Array.from({ length: Math.min(6, candidates.length) }, async () => {
    while (nextIndex < candidates.length && !signal.aborted) {
      const model = candidates[nextIndex++]
      if (!model) {
        continue
      }
      try {
        const response = await fetcher(showUrl(tagsUrl), {
          method: 'POST',
          signal,
          headers: { Accept: 'application/json', 'Content-Type': 'application/json' },
          body: JSON.stringify({ model: model.name })
        })
        if (!response.ok) {
          continue
        }
        const body = record(await response.json())
        const capabilities = Array.isArray(body?.capabilities)
          ? body.capabilities.filter(
              (capability): capability is string => typeof capability === 'string'
            )
          : []
        if (capabilities.includes('embedding') && !capabilities.includes('completion')) {
          excluded.add(model.name)
        }
      } catch {
        // Older Ollama versions may not expose capabilities.
      }
    }
  })
  await Promise.all(workers)
  return excluded
}
