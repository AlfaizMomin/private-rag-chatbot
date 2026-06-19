import {
  AfterViewChecked,
  ChangeDetectionStrategy,
  ChangeDetectorRef,
  Component,
  ElementRef,
  OnDestroy,
  OnInit,
  ViewChild,
} from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { Subscription } from 'rxjs';
import { ChatService } from './chat.service';

export interface ChatMessage {
  role: 'user' | 'bot';
  content: string;
  isStreaming: boolean;
  timestamp: Date;
}

@Component({
  selector: 'app-chat',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './chat.component.html',
  styleUrl: './chat.component.css',
  changeDetection: ChangeDetectionStrategy.Default,
})
export class ChatComponent implements OnInit, AfterViewChecked, OnDestroy {
  @ViewChild('scrollAnchor') private scrollAnchor!: ElementRef<HTMLDivElement>;
  @ViewChild('inputRef')     private inputRef!: ElementRef<HTMLTextAreaElement>;

  messages:    ChatMessage[] = [];
  inputText    = '';
  isStreaming  = false;
  backendReady = false;
  statusMsg    = 'Connecting to backend…';

  private subscription?: Subscription;
  private needsScroll   = false;

  constructor(
    private chatService: ChatService,
    private cdr: ChangeDetectorRef,
  ) {}

  async ngOnInit(): Promise<void> {
    await this.waitForBackend();
  }

  ngAfterViewChecked(): void {
    if (this.needsScroll) {
      this.scrollAnchor?.nativeElement.scrollIntoView({ behavior: 'instant' });
      this.needsScroll = false;
    }
  }

  ngOnDestroy(): void {
    this.subscription?.unsubscribe();
  }

  // ── Public actions ──────────────────────────────────────────────────────────

  sendMessage(): void {
    const text = this.inputText.trim();
    if (!text || this.isStreaming || !this.backendReady) return;

    this.inputText = '';
    this.resetTextareaHeight();

    // Push user message
    this.messages.push({
      role: 'user',
      content: text,
      isStreaming: false,
      timestamp: new Date(),
    });

    // Placeholder bot message — content grows as tokens arrive
    const botMsg: ChatMessage = {
      role: 'bot',
      content: '',
      isStreaming: true,
      timestamp: new Date(),
    };
    this.messages.push(botMsg);
    this.isStreaming  = true;
    this.needsScroll  = true;

    this.subscription = this.chatService.stream(text).subscribe({
      next: (token) => {
        botMsg.content  += token;
        this.needsScroll = true;
        this.cdr.detectChanges(); // push change detection during stream
      },
      error: (err: Error) => {
        botMsg.content   = `⚠️ Error: ${err.message}`;
        botMsg.isStreaming = false;
        this.isStreaming  = false;
        this.needsScroll  = true;
        this.cdr.detectChanges();
      },
      complete: () => {
        botMsg.isStreaming = false;
        this.isStreaming   = false;
        this.needsScroll   = true;
        this.cdr.detectChanges();
        this.focusInput();
      },
    });
  }

  onKeydown(event: KeyboardEvent): void {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      this.sendMessage();
    }
  }

  onInput(event: Event): void {
    const el = event.target as HTMLTextAreaElement;
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 140) + 'px';
  }

  clearChat(): void {
    if (this.isStreaming) return;
    this.messages = [];
  }

  // ── Private helpers ─────────────────────────────────────────────────────────

  private async waitForBackend(): Promise<void> {
    let attempts = 0;
    const maxAttempts = 30; // 30 × 2 s = 60 s max wait

    while (attempts < maxAttempts) {
      const ready = await this.chatService.checkHealth();
      if (ready) {
        this.backendReady = true;
        this.statusMsg    = '';
        this.cdr.detectChanges();
        this.focusInput();
        return;
      }
      attempts++;
      this.statusMsg = `Initializing RAG pipeline… (attempt ${attempts}/${maxAttempts})`;
      this.cdr.detectChanges();
      await this.delay(2000);
    }

    this.statusMsg = '⚠️ Backend is unavailable. Is the server running on port 3000?';
    this.cdr.detectChanges();
  }

  private focusInput(): void {
    setTimeout(() => this.inputRef?.nativeElement.focus(), 50);
  }

  private resetTextareaHeight(): void {
    if (this.inputRef?.nativeElement) {
      this.inputRef.nativeElement.style.height = 'auto';
    }
  }

  private delay(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  // Used by @for track expression
  trackByIndex(index: number): number {
    return index;
  }
}
