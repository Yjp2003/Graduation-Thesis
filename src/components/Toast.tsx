/**
 * Toast Notification System
 * Global event-driven toast manager with animated notifications.
 */
import React, { useState, useEffect, useCallback } from 'react';
import { CheckCircle2, AlertCircle, Info, AlertTriangle, X } from 'lucide-react';

export type ToastType = 'success' | 'error' | 'warning' | 'info';

export interface ToastItem {
  id: string;
  message: string;
  type: ToastType;
  duration: number;
}

// ── Global Event System ──
type ToastListener = (t: ToastItem) => void;
const listeners: Set<ToastListener> = new Set();

export const toast = {
  show(message: string, type: ToastType = 'info', duration = 3500) {
    const t: ToastItem = {
      id: Math.random().toString(36).substr(2, 9),
      message,
      type,
      duration,
    };
    listeners.forEach(fn => fn(t));
  },
  success(msg: string, dur?: number) { this.show(msg, 'success', dur); },
  error(msg: string, dur?: number) { this.show(msg, 'error', dur); },
  warning(msg: string, dur?: number) { this.show(msg, 'warning', dur); },
  info(msg: string, dur?: number) { this.show(msg, 'info', dur); },
};

// ── Toast Container ──
export function ToastContainer() {
  const [toasts, setToasts] = useState<ToastItem[]>([]);

  useEffect(() => {
    const handler: ToastListener = (t) => {
      setToasts(prev => [...prev, t]);
    };
    listeners.add(handler);
    return () => { listeners.delete(handler); };
  }, []);

  const dismiss = useCallback((id: string) => {
    setToasts(prev => prev.filter(t => t.id !== id));
  }, []);

  return (
    <div className="fixed top-5 right-5 z-[200] flex flex-col gap-3 pointer-events-none" style={{ maxWidth: '420px' }}>
      {toasts.map(t => (
        <SingleToast key={t.id} item={t} onDismiss={dismiss} />
      ))}
    </div>
  );
}

// ── Single Toast ──
function SingleToast({ item, onDismiss }: { item: ToastItem; onDismiss: (id: string) => void; key?: string }) {
  const [phase, setPhase] = useState<'enter' | 'idle' | 'exit'>('enter');

  useEffect(() => {
    // Enter animation completes after mounting
    requestAnimationFrame(() => setPhase('idle'));

    const timer = setTimeout(() => {
      setPhase('exit');
      setTimeout(() => onDismiss(item.id), 350);
    }, item.duration);

    return () => clearTimeout(timer);
  }, [item, onDismiss]);

  const handleClose = () => {
    setPhase('exit');
    setTimeout(() => onDismiss(item.id), 350);
  };

  const config: Record<ToastType, { icon: React.ReactNode; border: string; glow: string; bar: string }> = {
    success: {
      icon: <CheckCircle2 className="w-5 h-5 text-emerald-400 shrink-0" />,
      border: 'border-emerald-500/30',
      glow: 'shadow-emerald-500/10',
      bar: 'bg-emerald-400',
    },
    error: {
      icon: <AlertCircle className="w-5 h-5 text-red-400 shrink-0" />,
      border: 'border-red-500/30',
      glow: 'shadow-red-500/10',
      bar: 'bg-red-400',
    },
    warning: {
      icon: <AlertTriangle className="w-5 h-5 text-amber-400 shrink-0" />,
      border: 'border-amber-500/30',
      glow: 'shadow-amber-500/10',
      bar: 'bg-amber-400',
    },
    info: {
      icon: <Info className="w-5 h-5 text-sky-400 shrink-0" />,
      border: 'border-sky-500/30',
      glow: 'shadow-sky-500/10',
      bar: 'bg-sky-400',
    },
  };

  const c = config[item.type];

  return (
    <div
      className={`
        pointer-events-auto flex items-start gap-3 pl-4 pr-3 py-3.5
        bg-[#1a1f27]/95 backdrop-blur-xl border ${c.border} rounded-xl
        shadow-2xl ${c.glow} relative overflow-hidden
        transition-all duration-300 ease-out
        ${phase === 'enter' ? 'opacity-0 translate-x-8 scale-95' : ''}
        ${phase === 'idle' ? 'opacity-100 translate-x-0 scale-100' : ''}
        ${phase === 'exit' ? 'opacity-0 translate-x-8 scale-95' : ''}
      `}
    >
      {/* Progress bar */}
      <div className="absolute bottom-0 left-0 right-0 h-[2px] bg-white/5">
        <div
          className={`h-full ${c.bar} opacity-60`}
          style={{
            animation: `toast-progress ${item.duration}ms linear forwards`,
          }}
        />
      </div>

      {c.icon}
      <p className="text-sm text-slate-200 flex-1 leading-relaxed pt-0.5">{item.message}</p>
      <button
        onClick={handleClose}
        className="text-slate-500 hover:text-white transition-colors shrink-0 p-0.5 rounded hover:bg-white/10"
      >
        <X className="w-3.5 h-3.5" />
      </button>
    </div>
  );
}
