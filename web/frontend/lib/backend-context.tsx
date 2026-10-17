'use client';

import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';

export interface BackendNoticeOptions {
  title?: string;
  message?: string;
  actionType?: 'participant' | 'inference' | 'save' | 'general';
}

interface BackendContextType {
  isBackendConnected: boolean;
  isChecking: boolean;
  checkBackend: () => Promise<boolean>;
  showBackendModal: (options?: BackendNoticeOptions) => void;
  closeBackendModal: () => void;
  modalState: {
    isOpen: boolean;
    title: string;
    message: string;
    actionType?: string;
  };
}

const defaultContext: BackendContextType = {
  isBackendConnected: false,
  isChecking: true,
  checkBackend: async () => false,
  showBackendModal: () => {},
  closeBackendModal: () => {},
  modalState: {
    isOpen: false,
    title: 'Backend Service Unavailable',
    message: '',
  },
};

const BackendContext = createContext<BackendContextType>(defaultContext);

// Global listener for non-React call sites (e.g. inside api.ts)
let globalShowNoticeListener: ((options?: BackendNoticeOptions) => void) | null = null;

export function notifyBackendRequired(options?: BackendNoticeOptions) {
  if (globalShowNoticeListener) {
    globalShowNoticeListener(options);
  }
}

export function BackendProvider({ children }: { children: React.ReactNode }) {
  const [isBackendConnected, setIsBackendConnected] = useState<boolean>(false);
  const [isChecking, setIsChecking] = useState<boolean>(true);
  const [modalState, setModalState] = useState<{
    isOpen: boolean;
    title: string;
    message: string;
    actionType?: string;
  }>({
    isOpen: false,
    title: 'Backend Service Unavailable',
    message: '',
  });

  const checkBackend = useCallback(async (): Promise<boolean> => {
    setIsChecking(true);
    const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8765';
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 1200);

      const res = await fetch(`${apiUrl}/api/health`, {
        signal: controller.signal,
        headers: { Accept: 'application/json' },
      });
      clearTimeout(timeoutId);

      if (res.ok) {
        const data = await res.json();
        const connected = data && data.status === 'ok';
        setIsBackendConnected(connected);
        setIsChecking(false);
        return connected;
      }
      setIsBackendConnected(false);
      setIsChecking(false);
      return false;
    } catch {
      setIsBackendConnected(false);
      setIsChecking(false);
      return false;
    }
  }, []);

  useEffect(() => {
    let active = true;
    const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8765';
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 1200);

    fetch(`${apiUrl}/api/health`, {
      signal: controller.signal,
      headers: { Accept: 'application/json' },
    })
      .then((res) => {
        clearTimeout(timeoutId);
        return res.ok ? res.json() : null;
      })
      .then((data) => {
        if (!active) return;
        setIsBackendConnected(data?.status === 'ok');
        setIsChecking(false);
      })
      .catch(() => {
        if (!active) return;
        setIsBackendConnected(false);
        setIsChecking(false);
      });

    return () => {
      active = false;
      controller.abort();
    };
  }, []);

  const showBackendModal = useCallback((options?: BackendNoticeOptions) => {
    const defaultTitle = 'Backend Service Unavailable';
    let defaultMsg =
      'This hosted research demo is running in standalone mode without a connected backend. You can explore the included benchmark dataset (Participant sub-01) and interactive visualizations. For full cohort access (sub-01 to sub-05) and live compute, download or clone the repository and run the application locally with the backend configured.';

    if (options?.actionType === 'participant') {
      defaultMsg =
        'Accessing additional cohort participants (sub-02 through sub-15) requires the Python scientific backend and raw OpenNeuro dataset (ds006840). In this hosted demo, Participant sub-01 is fully available for exploration. Clone or download the repository to run all participants locally.';
    } else if (options?.actionType === 'inference') {
      defaultMsg =
        'Live model inference executes dynamic Ridge trajectory extrapolation and LDA spatial filtering on the Python scientific backend. In this hosted demo, deterministic trial replay and 3D kinematics are active for Participant sub-01. Run the backend locally to execute live model recomputations.';
    }

    setModalState({
      isOpen: true,
      title: options?.title || defaultTitle,
      message: options?.message || defaultMsg,
      actionType: options?.actionType || 'general',
    });
  }, []);

  const closeBackendModal = useCallback(() => {
    setModalState((prev) => ({ ...prev, isOpen: false }));
  }, []);

  useEffect(() => {
    globalShowNoticeListener = showBackendModal;
    return () => {
      globalShowNoticeListener = null;
    };
  }, [showBackendModal]);

  return (
    <BackendContext.Provider
      value={{
        isBackendConnected,
        isChecking,
        checkBackend,
        showBackendModal,
        closeBackendModal,
        modalState,
      }}
    >
      {children}
    </BackendContext.Provider>
  );
}

export function useBackend() {
  return useContext(BackendContext);
}
