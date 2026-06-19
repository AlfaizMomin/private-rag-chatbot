import 'dotenv/config';
import express from 'express';
import cors from 'cors';
import fs from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

import { ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings } from '@langchain/google-genai';
import { MemoryVectorStore } from 'langchain/vectorstores/memory';
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';
import { ChatPromptTemplate } from '@langchain/core/prompts';
import { StringOutputParser } from '@langchain/core/output_parsers';
import { RunnableSequence, RunnablePassthrough } from '@langchain/core/runnables';

// ── ES-module __dirname shim ──────────────────────────────────────────────────
const __filename = fileURLToPath(import.meta.url);
const __dirname  = dirname(__filename);

// ── App & config ──────────────────────────────────────────────────────────────
const app  = express();
const PORT = process.env.PORT         || 3000;
const RAW_GEMINI_KEY = process.env.GEMINI_API_KEY || '';
const RAW_GOOGLE_KEY = process.env.GOOGLE_API_KEY || '';
const GEMINI_API_KEY = (RAW_GEMINI_KEY || RAW_GOOGLE_KEY || '').trim();
const LLM_MODEL      = process.env.LLM_MODEL || 'gemini-2.5-flash';
const EMBED_MODEL    = (process.env.EMBED_MODEL || 'gemini-embedding-2').trim();
const CORS_ORIGIN    = process.env.CORS_ORIGIN || 'http://localhost:4200';
const USE_PLACEHOLDER_KEY = GEMINI_API_KEY === 'your_gemini_api_key_here' || GEMINI_API_KEY === 'YOUR_GEMINI_API_KEY';
const WHICH_KEY_ENV = RAW_GEMINI_KEY ? 'GEMINI_API_KEY' : RAW_GOOGLE_KEY ? 'GOOGLE_API_KEY' : 'none';

if (EMBED_MODEL === 'embedding-001' || EMBED_MODEL === 'text-embedding-004') {
  console.warn('[env-check] EMBED_MODEL=' + EMBED_MODEL + ' is not the correct Gemini embedding identifier for this API. Use gemini-embedding-001 or gemini-embedding-2.');
}

console.log('[env-check] Using API key source:', WHICH_KEY_ENV);
console.log('[env-check] LLM model:', LLM_MODEL);
console.log('[env-check] Embedding model:', EMBED_MODEL);
console.log('[env-check] CORS origin:', CORS_ORIGIN);

if (!GEMINI_API_KEY || USE_PLACEHOLDER_KEY) {
  console.warn('[env-check] No valid Gemini API key detected. The backend will fail during initialization.');
  console.warn('[env-check] Expected one of: GEMINI_API_KEY or GOOGLE_API_KEY');
}

app.use(cors({ origin: CORS_ORIGIN }));
app.use(express.json());

// ── System prompt — strict factuality ────────────────────────────────────────
// Temperature 0 + this prompt = deterministic, grounded answers only.
const SYSTEM_TEMPLATE = `You are a precise, factual assistant. \
Your ONLY source of truth is the CONTEXT block below.

STRICT RULES (never break these):
1. Answer using ONLY information explicitly present in the CONTEXT.
2. If the answer cannot be found in the CONTEXT, respond with EXACTLY this sentence and nothing else:
   "I'm sorry, but I cannot find that information in my provided data."
3. Never speculate, infer beyond what is written, or use outside knowledge.
4. Never fabricate names, numbers, dates, or facts.
5. Be concise and factually precise.

--- CONTEXT START ---
{context}
--- CONTEXT END ---`;

// ── RAG pipeline state ────────────────────────────────────────────────────────
let ragChain   = null;
let isReady    = false;
let initError  = null;

async function initializeRAG() {
  if (!GEMINI_API_KEY || USE_PLACEHOLDER_KEY) {
    const detail = `API key source=${WHICH_KEY_ENV}; LLM_MODEL=${LLM_MODEL}; EMBED_MODEL=${EMBED_MODEL}`;
    throw new Error(
      'Gemini API configuration is invalid. ' +
      `Env source: ${WHICH_KEY_ENV}. ` +
      `Please set a real GEMINI_API_KEY or GOOGLE_API_KEY in Render. ` +
      detail
    );
  }

  console.log('\n╔══════════════════════════════════════╗');
  console.log('║   Initializing RAG Pipeline          ║');
  console.log('╚══════════════════════════════════════╝\n');

  // ── 1. Load the knowledge file ───────────────────────────────────────────
  const knowledgePath = join(__dirname, 'knowledge', 'data.txt');
  if (!fs.existsSync(knowledgePath)) {
    throw new Error(
      `Knowledge file not found at: ${knowledgePath}\n` +
      `Please create backend/knowledge/data.txt with your content.`
    );
  }
  const rawText = fs.readFileSync(knowledgePath, 'utf-8');
  console.log(`📄  Knowledge file loaded  (${rawText.length.toLocaleString()} characters)`);

  // ── 2. Split into overlapping chunks ────────────────────────────────────
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize:    500,   // characters per chunk
    chunkOverlap: 60,    // overlap prevents context loss at boundaries
    separators: ['\n\n', '\n', '. ', '! ', '? ', ' ', ''],
  });
  const docs = await splitter.createDocuments([rawText]);
  console.log(`✂️   Chunked into ${docs.length} document segments`);

  // ── 3. Embeddings model (Gemini) ───────────────────────────────────────
  const embeddings = new GoogleGenerativeAIEmbeddings({
    model:   EMBED_MODEL,
    apiKey:  GEMINI_API_KEY,
  });

  // ── 4. In-memory vector store — indexed on startup ───────────────────────
  // Swap this for ChromaDB or Qdrant for persistence with large datasets.
  console.log(`🔢  Embedding ${docs.length} chunks with "${EMBED_MODEL}" …`);
  console.log('    (First run may take 30–60 s while the model warms up)\n');
  const vectorStore = await MemoryVectorStore.fromDocuments(docs, embeddings);
  const retriever   = vectorStore.asRetriever({
    k: 5,                     // retrieve top-5 most relevant chunks
    searchType: 'similarity',
  });
  console.log('✅  Vector store ready\n');

  // ── 5. LLM — Gemini via Google AI ───────────────────────────────────────
  const llm = new ChatGoogleGenerativeAI({
    model:          LLM_MODEL,
    apiKey:         GEMINI_API_KEY,
    temperature:    0,            // strict factuality — no creativity
    maxOutputTokens: 1024,       // max tokens per response
  });

  // ── 6. Prompt template ───────────────────────────────────────────────────
  const promptTemplate = ChatPromptTemplate.fromMessages([
    ['system', SYSTEM_TEMPLATE],
    ['human',  '{question}'],
  ]);

  // ── 7. LCEL RAG chain: retrieve → format → prompt → LLM → parse ─────────
  //   Input shape expected:  { question: string }
  //   Output:                async string token stream
  ragChain = RunnableSequence.from([
    RunnablePassthrough.assign({
      context: async (input) => {
        const retrieved = await retriever.invoke(input.question);
        return retrieved
          .map((doc, i) => `[Chunk ${i + 1}]\n${doc.pageContent}`)
          .join('\n\n---\n\n');
      },
    }),
    promptTemplate,
    llm,
    new StringOutputParser(),
  ]);

  isReady = true;
  console.log('🎯  RAG chain assembled and ready\n');
}

// ── Routes ────────────────────────────────────────────────────────────────────

// Health-check — Angular app polls this on startup
app.get('/api/health', (_req, res) => {
  res.setHeader('Cache-Control', 'no-store, no-cache, must-revalidate, proxy-revalidate');
  res.setHeader('Pragma', 'no-cache');
  res.setHeader('Expires', '0');

  res.json({
    status:   isReady ? 'ready' : initError ? 'error' : 'initializing',
    provider: 'gemini',
    model:    LLM_MODEL,
    embedder: EMBED_MODEL,
    error:    initError?.message ?? null,
  });
});

// Streaming chat endpoint — Server-Sent Events (SSE)
app.post('/api/chat/stream', async (req, res) => {
  // Guard: pipeline must be ready
  if (!isReady) {
    const code = initError ? 500 : 503;
    return res.status(code).json({
      error: initError
        ? `Initialization failed: ${initError.message}`
        : 'RAG pipeline is still initializing. Please retry in a moment.',
    });
  }

  const { message } = req.body;
  if (!message?.trim()) {
    return res.status(400).json({ error: 'message field is required and must not be empty.' });
  }

  // ── SSE response headers ─────────────────────────────────────────────────
  res.setHeader('Content-Type',  'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection',    'keep-alive');
  res.setHeader('X-Accel-Buffering', 'no'); // Disable nginx buffering if proxied
  res.flushHeaders();

  // Heartbeat keeps the connection alive through load balancers
  const heartbeat = setInterval(() => res.write(': heartbeat\n\n'), 15_000);

  const cleanup = () => {
    clearInterval(heartbeat);
    if (!res.writableEnded) {
      res.end();
    }
  };

  // Abort stream if client disconnects early
  res.on('close', cleanup);

  try {
    const tokenStream = await ragChain.stream({ question: message.trim() });

    for await (const token of tokenStream) {
      if (token) {
        // Each SSE message: "data: <json>\n\n"
        res.write(`data: ${JSON.stringify({ token })}\n\n`);
      }
    }

    // Signal end-of-stream to the frontend
    res.write('data: [DONE]\n\n');
  } catch (err) {
    console.error('❌  Stream error:', err.message);
    res.write(`data: ${JSON.stringify({ error: err.message })}\n\n`);
  } finally {
    cleanup();
  }
});

function startServer(port) {
  const server = app.listen(port, () => {
    console.log(`🌐  Server listening on http://localhost:${port}`);
    console.log(`📡  Stream endpoint: POST http://localhost:${port}/api/chat/stream`);
    console.log(`❤️   Health check:    GET  http://localhost:${port}/api/health\n`);
  });

  server.on('error', (err) => {
    if (err.code === 'EADDRINUSE') {
      const fallbackPort = port + 1;
      console.warn(`⚠️  Port ${port} is busy, retrying on ${fallbackPort}...`);
      startServer(fallbackPort);
      return;
    }

    console.error('❌  Server startup error:', err.message);
  });
}

// ── Bootstrap ─────────────────────────────────────────────────────────────────
initializeRAG()
  .then(() => {
    startServer(PORT);
  })
  .catch((err) => {
    initError = err;
    console.error('\n❌  RAG initialization failed:', err.message);
    // Still start the server so the health endpoint reports the error
    startServer(PORT);
  });
