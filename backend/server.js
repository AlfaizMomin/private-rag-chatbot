import 'dotenv/config';
import express from 'express';
import cors from 'cors';
import fs from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

import { ChatOllama, OllamaEmbeddings } from '@langchain/ollama';
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
const OLLAMA_BASE_URL = process.env.OLLAMA_BASE_URL || 'http://localhost:11434';
const LLM_MODEL       = process.env.LLM_MODEL       || 'llama3.2';
const EMBED_MODEL     = process.env.EMBED_MODEL      || 'nomic-embed-text';
const CORS_ORIGIN     = process.env.CORS_ORIGIN      || 'http://localhost:4200';

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

  // ── 3. Embeddings model (nomic-embed-text via Ollama) ───────────────────
  const embeddings = new OllamaEmbeddings({
    model:   EMBED_MODEL,
    baseUrl: OLLAMA_BASE_URL,
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

  // ── 5. LLM — llama3.2 via Ollama ─────────────────────────────────────────
  const llm = new ChatOllama({
    model:       LLM_MODEL,
    baseUrl:     OLLAMA_BASE_URL,
    temperature: 0,            // strict factuality — no creativity
    numPredict:  1024,         // max tokens per response
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
  res.json({
    status:   isReady ? 'ready' : initError ? 'error' : 'initializing',
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
    res.end();
  };

  // Abort stream if client disconnects early
  req.on('close', cleanup);

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

// ── Bootstrap ─────────────────────────────────────────────────────────────────
initializeRAG()
  .then(() => {
    app.listen(PORT, () => {
      console.log(`🌐  Server listening on http://localhost:${PORT}`);
      console.log(`📡  Stream endpoint: POST http://localhost:${PORT}/api/chat/stream`);
      console.log(`❤️   Health check:    GET  http://localhost:${PORT}/api/health\n`);
    });
  })
  .catch((err) => {
    initError = err;
    console.error('\n❌  RAG initialization failed:', err.message);
    // Still start the server so the health endpoint reports the error
    app.listen(PORT, () => {
      console.log(`⚠️   Server started in degraded mode on http://localhost:${PORT}`);
    });
  });
