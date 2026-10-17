'use client';

import React, { useState, useEffect } from 'react';
import {
  ServerOff,
  X,
  Copy,
  Check,
  ExternalLink,
  Terminal,
  Sparkles,
} from 'lucide-react';
import { useBackend } from '@/lib/backend-context';

export default function BackendNoticeModal() {
  const { modalState, closeBackendModal } = useBackend();
  const [copied, setCopied] = useState(false);

  const cloneCommand = `./run_project.sh`;

  const handleCopy = () => {
    navigator.clipboard.writeText(cloneCommand);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && modalState.isOpen) {
        closeBackendModal();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [modalState.isOpen, closeBackendModal]);

  if (!modalState.isOpen) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4 sm:p-6 overflow-y-auto animate-in fade-in duration-200"
      role="dialog"
      aria-modal="true"
      aria-labelledby="backend-modal-title"
    >
      {/* Backdrop */}
      <div
        className="fixed inset-0 bg-[#1C1C1A]/45 backdrop-blur-sm transition-opacity"
        onClick={closeBackendModal}
      />

      {/* Modal Dialog Card */}
      <div className="relative w-full max-w-lg bg-[#FFFFFF] border border-[#D9D8D0] rounded-3xl p-6 sm:p-7 shadow-2xl space-y-5 font-body z-10">
        {/* Header Bar */}
        <div className="flex items-start justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-2xl bg-[#FEF3C7] border border-[#FDE68A] flex items-center justify-center text-[#B45309]">
              <ServerOff className="w-5 h-5" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="text-[10px] font-bold tracking-[0.25em] uppercase text-[#B45309] bg-[#FEF3C7] px-2 py-0.5 rounded-full border border-[#FDE68A]">
                  Hosted Demo Notice
                </span>
              </div>
              <h3
                id="backend-modal-title"
                className="font-display text-lg font-bold text-[#1C1C1A] mt-1"
              >
                {modalState.title}
              </h3>
            </div>
          </div>

          <button
            onClick={closeBackendModal}
            className="p-1.5 rounded-full text-[#6B6B66] hover:text-[#1C1C1A] hover:bg-[#EDECE6] transition-colors cursor-pointer"
            aria-label="Close modal"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Informational Message */}
        <div className="text-sm text-[#4A4A46] leading-relaxed bg-[#FAF9F5] border border-[#EDECE6] rounded-2xl p-4 space-y-2">
          <p>{modalState.message}</p>
          <div className="flex items-center gap-1.5 text-xs font-semibold text-[#3A5244] pt-1">
            <Sparkles className="w-3.5 h-3.5 text-[#3A5244]" />
            <span>Participant sub-01 benchmark trials and full 3D kinematics remain fully interactive.</span>
          </div>
        </div>

        {/* Local Run Instructions */}
        <div className="space-y-2">
          <label className="block text-xs font-semibold text-[#1C1C1A] flex items-center justify-between">
            <span className="flex items-center gap-1.5">
              <Terminal className="w-3.5 h-3.5 text-[#3A5244]" />
              Run Full Stack Locally (All 5 Participants + Live Compute)
            </span>
          </label>

          <div className="flex items-center justify-between bg-[#1C1C1A] text-[#F5F3EE] rounded-xl px-3.5 py-2.5 font-mono text-xs shadow-inner">
            <div className="flex items-center gap-2 overflow-x-auto select-all">
              <span className="text-[#AEADA4] select-none">$</span>
              <span>{cloneCommand}</span>
            </div>
            <button
              onClick={handleCopy}
              className="ml-2 px-2.5 py-1 rounded-lg bg-[#2A2A28] hover:bg-[#3A3A38] text-xs font-body text-[#F5F3EE] flex items-center gap-1 transition-colors cursor-pointer shrink-0"
              title="Copy command"
            >
              {copied ? (
                <>
                  <Check className="w-3 h-3 text-[#10B981]" />
                  <span className="text-[#10B981]">Copied</span>
                </>
              ) : (
                <>
                  <Copy className="w-3 h-3" />
                  <span>Copy</span>
                </>
              )}
            </button>
          </div>
          <p className="text-[11px] text-[#6B6B66] leading-normal">
            Starts the Python FastAPI backend (<code className="text-[#3A5244]">port 8765</code>) and Next.js frontend with full cohort access.
          </p>
        </div>

        {/* Action Buttons */}
        <div className="flex flex-col sm:flex-row gap-2.5 pt-1">
          <button
            onClick={closeBackendModal}
            className="flex-1 inline-flex items-center justify-center gap-2 px-5 py-2.5 rounded-full text-xs font-semibold bg-[#1C1C1A] text-[#F5F3EE] hover:bg-[#3A5244] transition-colors cursor-pointer shadow-xs"
          >
            Continue with sub-01 Demo
          </button>
          <a
            href="https://github.com/omkarparkhe111/Brain_Movement"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center justify-center gap-1.5 px-4 py-2.5 rounded-full text-xs font-semibold border border-[#D9D8D0] hover:bg-[#EDECE6] text-[#1C1C1A] transition-colors cursor-pointer"
          >
            <span>Repository</span>
            <ExternalLink className="w-3.5 h-3.5 opacity-60" />
          </a>
        </div>
      </div>
    </div>
  );
}
