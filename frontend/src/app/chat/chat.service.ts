import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';
import { environment } from '../../environment';

export interface StreamError {
  error: string;
}

@Injectable({ providedIn: 'root' })
export class ChatService {
  private readonly endpoint =
  `${environment.apiBaseUrl}/chat/stream`;

  /**
   * Opens an SSE stream to the backend RAG endpoint.
   * Emits each token string as it arrives, then completes.
   * The observable tears down the fetch (AbortController) on unsubscribe.
   */
  stream(message: string): Observable<string> {
    return new Observable<string>((observer) => {
      const controller = new AbortController();
      let completed = false;

      fetch(this.endpoint, {
        method: 'POST',
        cache: 'no-store',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message }),
        signal: controller.signal,
      })
        .then(async (response) => {
          if (!response.ok) {
            const body = await response.text().catch(() => '');
            throw new Error(`HTTP ${response.status}: ${body || response.statusText}`);
          }

          if (!response.body) {
            throw new Error('The server returned an empty response body.');
          }

          const reader  = response.body.getReader();
          const decoder = new TextDecoder();
          let buffer    = '';

          // Read the raw byte stream chunk by chunk
          while (true) {
            const { done, value } = await reader.read();

            if (done) break;

            // Decode this chunk and append to our line buffer
            buffer += decoder.decode(value, { stream: true });

            // SSE messages are separated by double newlines.
            // We split on \n and process complete lines, keeping
            // any partial (incomplete) line in the buffer.
            const lines = buffer.split('\n');
            buffer = lines.pop() ?? ''; // last element may be incomplete

            for (const line of lines) {
              const trimmed = line.trim();

              // Skip SSE comments (heartbeat lines starting with ':')
              if (!trimmed || trimmed.startsWith(':')) continue;

              if (trimmed.startsWith('data: ')) {
                const payload = trimmed.slice(6); // strip "data: " prefix

                // Backend sends [DONE] as the end-of-stream sentinel
                if (payload === '[DONE]') {
                  completed = true;
                  observer.complete();
                  return;
                }

                try {
                  const parsed = JSON.parse(payload) as { token?: string; error?: string };

                  if (parsed.error) {
                    observer.error(new Error(parsed.error));
                    return;
                  }

                  if (parsed.token != null) {
                    observer.next(parsed.token);
                  }
                } catch {
                  // Silently skip malformed JSON frames
                }
              }
            }
          }

          if (!completed) observer.complete();
        })
        .catch((err: Error) => {
          if (err.name !== 'AbortError') {
            observer.error(err);
          }
        });

      // Teardown: abort the in-flight fetch when Angular unsubscribes
      return () => controller.abort();
    });
  }

  /** Checks whether the backend is ready before the user sends a message. */
  async checkHealth(): Promise<boolean> {
    try {
      const res = await fetch(`${environment.apiBaseUrl}/health`, {
        cache: 'no-store',
        headers: {
          'Cache-Control': 'no-cache',
          Pragma: 'no-cache',
        },
      });

      if (res.status === 304) {
        return true;
      }

      if (!res.ok) {
        return false;
      }

      const data = await res.json() as { status?: string };
      return data.status === 'ready';
    } catch {
      return false;
    }
  }
}
