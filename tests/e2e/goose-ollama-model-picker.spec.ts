import { mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { test, expect } from './helpers/orca-app'
import { waitForActiveWorktree, waitForSessionReady } from './helpers/store'

const gooseConfigDir = mkdtempSync(path.join(os.tmpdir(), 'orca-goose-e2e-'))
writeFileSync(
  path.join(gooseConfigDir, 'config.yaml'),
  [
    'GOOSE_PROVIDER: ollama',
    'GOOSE_MODEL: llama3.1:8b',
    'OLLAMA_HOST: http://localhost:11434',
    ''
  ].join('\n')
)
test.use({
  orcaAppExtraEnv: {
    GOOSE_CONFIG_DIR: gooseConfigDir,
    GOOSE_PROVIDER: 'ollama',
    GOOSE_MODEL: 'llama3.1:8b',
    OLLAMA_HOST: 'http://localhost:11434'
  }
})
test.afterAll(() => rmSync(gooseConfigDir, { recursive: true, force: true }))

test('Goose native chat lists and changes local Ollama models', async ({ orcaPage }) => {
  test.skip(
    process.env.ORCA_E2E_LIVE_OLLAMA !== '1',
    'Requires Goose ACP and a running local Ollama instance'
  )
  await waitForSessionReady(orcaPage)
  await waitForActiveWorktree(orcaPage)
  await orcaPage.evaluate(async () => {
    const settings = await window.api.settings.set({ experimentalNativeChat: true })
    window.__store?.setState({ settings })
  })

  await orcaPage.getByRole('button', { name: 'New tab', exact: true }).click()
  await orcaPage.getByRole('menuitem', { name: 'Goose', exact: true }).click()

  const composer = orcaPage.getByRole('textbox', {
    name: 'Ask anything, @ to mention files, / for commands',
    exact: true
  })
  await expect(composer).toBeVisible({ timeout: 60_000 })
  await composer.fill('Reply with exactly OK. Do not use tools or change files.')
  await orcaPage.getByRole('button', { name: 'Send', exact: true }).click()
  await expect(orcaPage.getByRole('button', { name: /^Done goose Chat/ })).toBeVisible({
    timeout: 120_000
  })

  const modelPicker = orcaPage.getByRole('button', { name: /^Model / })
  await expect(modelPicker).toContainText('llama3.1', { timeout: 60_000 })
  await modelPicker.click()

  const choices = orcaPage.locator('[data-slot="popover-content"] [role="option"]')
  await expect.poll(() => choices.count(), { timeout: 30_000 }).toBeGreaterThan(1)
  const labels = await choices.allTextContents()
  expect(labels.join(' ')).not.toMatch(/bge-m3|nomic-embed-text/i)
  const currentModelIndex = labels.findIndex((label) => label.includes('llama3.1'))
  const nextModelIndex = labels.findIndex((label) => !label.includes('llama3.1'))
  expect(currentModelIndex).toBeGreaterThanOrEqual(0)
  expect(nextModelIndex).toBeGreaterThanOrEqual(0)
  const selectedModel = await choices.nth(nextModelIndex).getAttribute('data-value')
  if (!selectedModel) {
    throw new Error('The alternate model option had no model identifier')
  }
  await choices.nth(nextModelIndex).click()
  await expect(modelPicker).not.toContainText('llama3.1')
  await expect
    .poll(() =>
      orcaPage.evaluate(
        () => window.__store?.getState().settings?.nativeChatSessionOptions?.goose?.model ?? null
      )
    )
    .toBe(selectedModel)
})
