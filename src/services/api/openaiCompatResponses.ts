/**
 * OpenAI-compatible Responses API streaming client (Stage 1: text-only).
 *
 * Activated when `CLAUDE_CODE_OPENAI_COMPAT=1` is set.
 * Required env vars: OPENAI_BASE_URL, OPENAI_API_KEY, OPENAI_MODEL.
 *
 * NOTE: Tool / function calling is NOT implemented in Stage 1.
 *       Only plain text chat streaming is supported.
 */

import type { Message } from '../../types/message.js'

/** A validated set of config values read from env vars. */
export interface OpenAICompatConfig {
  baseUrl: string
  apiKey: string
  model: string
}

/** A single text-delta yielded by the streaming generator. */
export interface TextDelta {
  text: string
}

/**
 * Read and validate required environment variables.
 * Throws a descriptive error when any required variable is missing.
 */
export function getOpenAICompatConfig(): OpenAICompatConfig {
  const baseUrl = process.env.OPENAI_BASE_URL?.replace(/\/$/, '')
  const apiKey = process.env.OPENAI_API_KEY
  const model = process.env.OPENAI_MODEL

  const missing: string[] = []
  if (!baseUrl) missing.push('OPENAI_BASE_URL')
  if (!apiKey) missing.push('OPENAI_API_KEY')
  if (!model) missing.push('OPENAI_MODEL')

  if (missing.length > 0) {
    throw new Error(
      `CLAUDE_CODE_OPENAI_COMPAT is enabled but the following required environment ` +
        `variables are not set: ${missing.join(', ')}. ` +
        `Please set them before running Claude Code with OpenAI-compatible mode.`,
    )
  }

  return { baseUrl: baseUrl!, apiKey: apiKey!, model: model! }
}

/**
 * Convert Claude Code's internal `Message[]` representation to the
 * Responses API `input` array format (text-only, Stage 1).
 *
 * - User messages: extract plain text from text-type content blocks.
 * - Assistant messages: extract plain text; ignore tool_use blocks.
 * - Messages with no extractable text are omitted.
 */
export function convertMessagesToResponsesInput(
  messages: Message[],
): Array<{ role: 'user' | 'assistant'; content: string }> {
  const result: Array<{ role: 'user' | 'assistant'; content: string }> = []

  for (const msg of messages) {
    if (msg.type !== 'user' && msg.type !== 'assistant') {
      continue
    }

    const role = msg.type === 'user' ? 'user' : 'assistant'
    const content = msg.message.content

    let text: string

    if (typeof content === 'string') {
      text = content
    } else if (Array.isArray(content)) {
      // Extract text from all text-type blocks; ignore everything else
      text = content
        .filter((block): block is { type: 'text'; text: string } =>
          typeof block === 'object' && block !== null && block.type === 'text',
        )
        .map(block => block.text)
        .join('\n')
    } else {
      continue
    }

    const trimmed = text.trim()
    if (trimmed.length > 0) {
      result.push({ role, content: trimmed })
    }
  }

  return result
}

/**
 * Parse a Server-Sent Events stream from the Responses API and yield each
 * incremental text delta.
 *
 * The Responses API uses the following SSE event types for text output:
 *   - `response.output_text.delta`  → contains `delta: "<text>"`
 *   - `response.completed`          → stream complete (stop iterating)
 *   - `error`                        → fatal error (throw)
 *
 * @yields {TextDelta} Incremental text deltas as they arrive.
 */
export async function* parseResponsesAPIStream(
  body: ReadableStream<Uint8Array>,
  signal: AbortSignal,
): AsyncGenerator<TextDelta, void> {
  const reader = body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  try {
    while (true) {
      if (signal.aborted) {
        break
      }

      const { done, value } = await reader.read()
      if (done) {
        break
      }

      buffer += decoder.decode(value, { stream: true })

      // Split by double-newline (SSE event separator)
      const events = buffer.split('\n\n')
      // Keep the last (possibly incomplete) chunk in the buffer
      buffer = events.pop() ?? ''

      for (const rawEvent of events) {
        const lines = rawEvent.split('\n')
        let eventType = ''
        let dataLine = ''

        for (const line of lines) {
          if (line.startsWith('event:')) {
            eventType = line.slice('event:'.length).trim()
          } else if (line.startsWith('data:')) {
            dataLine = line.slice('data:'.length).trim()
          }
        }

        if (!dataLine || dataLine === '[DONE]') {
          continue
        }

        let parsed: Record<string, unknown>
        try {
          parsed = JSON.parse(dataLine) as Record<string, unknown>
        } catch {
          // Malformed JSON — skip silently
          continue
        }

        // Normalize: use the `type` field inside the payload when the
        // outer SSE `event:` line is absent (some proxies omit it).
        const type =
          eventType || (typeof parsed.type === 'string' ? parsed.type : '')

        if (
          type === 'response.output_text.delta' ||
          (type === 'response.content_part.delta' &&
            parsed.delta !== undefined &&
            typeof (parsed.delta as Record<string, unknown>).text === 'string')
        ) {
          // response.output_text.delta → { delta: "<text>" }
          // response.content_part.delta → { delta: { text: "<text>" } }
          let delta: string | undefined

          if (typeof parsed.delta === 'string') {
            delta = parsed.delta
          } else if (
            parsed.delta !== null &&
            typeof parsed.delta === 'object' &&
            typeof (parsed.delta as Record<string, unknown>).text === 'string'
          ) {
            delta = (parsed.delta as Record<string, unknown>).text as string
          }

          if (delta && delta.length > 0) {
            yield { text: delta }
          }
        } else if (type === 'error') {
          const message =
            typeof parsed.message === 'string'
              ? parsed.message
              : typeof (parsed as Record<string, unknown>).error === 'string'
                ? ((parsed as Record<string, unknown>).error as string)
                : JSON.stringify(parsed)
          throw new Error(`OpenAI-compatible API error: ${message}`)
        } else if (
          type === 'response.completed' ||
          type === 'response.done' ||
          type === 'done'
        ) {
          return
        }
      }
    }
  } finally {
    reader.releaseLock()
  }
}

/**
 * Call the Responses API with streaming and yield incremental text deltas.
 *
 * @param config   Validated config (baseUrl, apiKey, model).
 * @param input    Converted input messages.
 * @param instructions  Optional system-prompt text (maps to `instructions`).
 * @param signal   AbortSignal forwarded from Claude Code's query infrastructure.
 * @yields {TextDelta} Incremental text deltas.
 */
export async function* streamResponsesAPI(
  config: OpenAICompatConfig,
  input: Array<{ role: 'user' | 'assistant'; content: string }>,
  instructions: string | undefined,
  signal: AbortSignal,
): AsyncGenerator<TextDelta, void> {
  const url = `${config.baseUrl}/v1/responses`

  const requestBody: Record<string, unknown> = {
    model: config.model,
    input,
    stream: true,
  }
  if (instructions && instructions.trim().length > 0) {
    requestBody.instructions = instructions
  }

  const response = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${config.apiKey}`,
      Accept: 'text/event-stream',
    },
    body: JSON.stringify(requestBody),
    signal,
  })

  if (!response.ok) {
    let errorBody = ''
    try {
      errorBody = await response.text()
    } catch {
      // ignore read errors
    }
    throw new Error(
      `OpenAI-compatible Responses API request failed with status ${response.status}: ${errorBody}`,
    )
  }

  if (!response.body) {
    throw new Error(
      'OpenAI-compatible Responses API returned an empty response body.',
    )
  }

  yield* parseResponsesAPIStream(response.body, signal)
}
