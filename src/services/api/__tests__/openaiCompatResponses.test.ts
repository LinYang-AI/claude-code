/**
 * Unit tests for the OpenAI-compatible Responses API SSE parser and helpers.
 *
 * These tests exercise `parseResponsesAPIStream`, `convertMessagesToResponsesInput`,
 * and `getOpenAICompatConfig` in isolation — no network calls are made.
 *
 * Run with Bun:
 *   bun test src/services/api/__tests__/openaiCompatResponses.test.ts
 */

import {
  convertMessagesToResponsesInput,
  getOpenAICompatConfig,
  parseResponsesAPIStream,
} from '../openaiCompatResponses.js'

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Encode a string as a UTF-8 ReadableStream. */
function toStream(text: string): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder()
  return new ReadableStream({
    start(controller) {
      controller.enqueue(encoder.encode(text))
      controller.close()
    },
  })
}

/** Collect all yielded TextDelta texts from the parser. */
async function collectDeltas(
  stream: ReadableStream<Uint8Array>,
  signal?: AbortSignal,
): Promise<string[]> {
  const deltas: string[] = []
  const abortCtrl = signal
    ? { signal }
    : { signal: new AbortController().signal }
  for await (const delta of parseResponsesAPIStream(
    stream,
    abortCtrl.signal,
  )) {
    deltas.push(delta.text)
  }
  return deltas
}

// ---------------------------------------------------------------------------
// parseResponsesAPIStream
// ---------------------------------------------------------------------------

describe('parseResponsesAPIStream', () => {
  test('parses response.output_text.delta events', async () => {
    const sse = [
      'event: response.output_text.delta',
      'data: {"type":"response.output_text.delta","delta":"Hello"}',
      '',
      'event: response.output_text.delta',
      'data: {"type":"response.output_text.delta","delta":", world"}',
      '',
      'event: response.completed',
      'data: {"type":"response.completed"}',
      '',
    ].join('\n')

    const deltas = await collectDeltas(toStream(sse))
    expect(deltas).toEqual(['Hello', ', world'])
  })

  test('handles response.content_part.delta with nested text object', async () => {
    const sse = [
      'event: response.content_part.delta',
      'data: {"type":"response.content_part.delta","delta":{"text":"chunk"}}',
      '',
      'event: response.done',
      'data: {"type":"response.done"}',
      '',
    ].join('\n')

    const deltas = await collectDeltas(toStream(sse))
    expect(deltas).toEqual(['chunk'])
  })

  test('stops at [DONE] sentinel', async () => {
    const sse = [
      'event: response.output_text.delta',
      'data: {"type":"response.output_text.delta","delta":"text"}',
      '',
      'data: [DONE]',
      '',
    ].join('\n')

    const deltas = await collectDeltas(toStream(sse))
    expect(deltas).toEqual(['text'])
  })

  test('skips empty delta strings', async () => {
    const sse = [
      'event: response.output_text.delta',
      'data: {"type":"response.output_text.delta","delta":""}',
      '',
      'event: response.output_text.delta',
      'data: {"type":"response.output_text.delta","delta":"real text"}',
      '',
      'event: response.completed',
      'data: {}',
      '',
    ].join('\n')

    const deltas = await collectDeltas(toStream(sse))
    expect(deltas).toEqual(['real text'])
  })

  test('throws on error events', async () => {
    const sse = [
      'event: error',
      'data: {"type":"error","message":"Rate limit exceeded"}',
      '',
    ].join('\n')

    await expect(collectDeltas(toStream(sse))).rejects.toThrow(
      'Rate limit exceeded',
    )
  })

  test('skips malformed JSON lines without throwing', async () => {
    const sse = [
      'data: not-valid-json',
      '',
      'event: response.output_text.delta',
      'data: {"type":"response.output_text.delta","delta":"ok"}',
      '',
      'event: response.completed',
      'data: {}',
      '',
    ].join('\n')

    const deltas = await collectDeltas(toStream(sse))
    expect(deltas).toEqual(['ok'])
  })

  test('handles type from data payload when event line is absent', async () => {
    const sse = [
      'data: {"type":"response.output_text.delta","delta":"no event line"}',
      '',
      'data: {"type":"response.completed"}',
      '',
    ].join('\n')

    const deltas = await collectDeltas(toStream(sse))
    expect(deltas).toEqual(['no event line'])
  })

  test('handles chunked delivery (split across reads)', async () => {
    // Simulate a stream where SSE events are split across multiple reads
    const part1 = 'event: response.output_text.delta\ndata: {"type":"response.output_text.'
    const part2 = 'delta","delta":"split"}\n\nevent: response.completed\ndata: {}\n\n'

    const encoder = new TextEncoder()
    const stream = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(encoder.encode(part1))
        controller.enqueue(encoder.encode(part2))
        controller.close()
      },
    })

    const deltas = await collectDeltas(stream)
    expect(deltas).toEqual(['split'])
  })
})

// ---------------------------------------------------------------------------
// convertMessagesToResponsesInput
// ---------------------------------------------------------------------------

describe('convertMessagesToResponsesInput', () => {
  test('converts plain user message', () => {
    const messages = [
      {
        type: 'user' as const,
        message: { role: 'user' as const, content: 'Hello!' },
        uuid: 'u1',
        timestamp: new Date().toISOString(),
      },
    ]
    const result = convertMessagesToResponsesInput(messages as any)
    expect(result).toEqual([{ role: 'user', content: 'Hello!' }])
  })

  test('extracts text from content block array', () => {
    const messages = [
      {
        type: 'user' as const,
        message: {
          role: 'user' as const,
          content: [
            { type: 'text', text: 'block one' },
            { type: 'image', source: { type: 'url', url: 'http://example.com' } }, // ignored
            { type: 'text', text: 'block two' },
          ],
        },
        uuid: 'u1',
        timestamp: new Date().toISOString(),
      },
    ]
    const result = convertMessagesToResponsesInput(messages as any)
    expect(result).toEqual([{ role: 'user', content: 'block one\nblock two' }])
  })

  test('ignores tool_use blocks from assistant messages', () => {
    const messages = [
      {
        type: 'assistant' as const,
        message: {
          role: 'assistant' as const,
          content: [
            { type: 'text', text: 'I will now call a tool.' },
            { type: 'tool_use', id: 'tool-1', name: 'bash', input: {} }, // ignored
          ],
        },
        uuid: 'a1',
        timestamp: new Date().toISOString(),
      },
    ] as any
    const result = convertMessagesToResponsesInput(messages)
    expect(result).toEqual([
      { role: 'assistant', content: 'I will now call a tool.' },
    ])
  })

  test('omits messages with no extractable text', () => {
    const messages = [
      {
        type: 'user' as const,
        message: {
          role: 'user' as const,
          content: [{ type: 'image', source: {} }], // no text
        },
        uuid: 'u1',
        timestamp: new Date().toISOString(),
      },
    ] as any
    const result = convertMessagesToResponsesInput(messages)
    expect(result).toEqual([])
  })

  test('skips non-user/non-assistant message types', () => {
    const messages = [
      { type: 'system', message: { content: 'system msg' } },
      {
        type: 'user' as const,
        message: { role: 'user' as const, content: 'hi' },
        uuid: 'u1',
        timestamp: new Date().toISOString(),
      },
    ] as any
    const result = convertMessagesToResponsesInput(messages)
    expect(result).toEqual([{ role: 'user', content: 'hi' }])
  })
})

// ---------------------------------------------------------------------------
// getOpenAICompatConfig
// ---------------------------------------------------------------------------

describe('getOpenAICompatConfig', () => {
  const originalEnv = { ...process.env }

  afterEach(() => {
    // Restore env after each test
    for (const key of ['OPENAI_BASE_URL', 'OPENAI_API_KEY', 'OPENAI_MODEL']) {
      if (key in originalEnv) {
        process.env[key] = originalEnv[key]
      } else {
        delete process.env[key]
      }
    }
  })

  test('returns config when all vars are set', () => {
    process.env.OPENAI_BASE_URL = 'https://api.deepseek.com/'
    process.env.OPENAI_API_KEY = 'sk-test'
    process.env.OPENAI_MODEL = 'deepseek-chat'

    const config = getOpenAICompatConfig()
    expect(config.baseUrl).toBe('https://api.deepseek.com') // trailing slash stripped
    expect(config.apiKey).toBe('sk-test')
    expect(config.model).toBe('deepseek-chat')
  })

  test('throws when OPENAI_BASE_URL is missing', () => {
    delete process.env.OPENAI_BASE_URL
    process.env.OPENAI_API_KEY = 'sk-test'
    process.env.OPENAI_MODEL = 'deepseek-chat'

    expect(() => getOpenAICompatConfig()).toThrow('OPENAI_BASE_URL')
  })

  test('throws listing all missing variables', () => {
    delete process.env.OPENAI_BASE_URL
    delete process.env.OPENAI_API_KEY
    delete process.env.OPENAI_MODEL

    let errorMsg = ''
    try {
      getOpenAICompatConfig()
    } catch (err) {
      errorMsg = (err as Error).message
    }
    expect(errorMsg).toContain('OPENAI_BASE_URL')
    expect(errorMsg).toContain('OPENAI_API_KEY')
    expect(errorMsg).toContain('OPENAI_MODEL')
  })
})
