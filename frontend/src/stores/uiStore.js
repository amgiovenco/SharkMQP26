import { create } from 'zustand';
import { devtools } from 'zustand/middleware';

// UI State Store (for modal/tab state if needed across components)
export const useUiStore = create(
    devtools((set) => ({
        isUploadModalOpen: false,
        isResultsModalOpen: false,
        activeTab: 'upload', // 'upload' | 'history' | 'settings'

        setUploadModalOpen: (open) => set({ isUploadModalOpen: open }),
        setResultsModalOpen: (open) => set({ isResultsModalOpen: open }),
        setActiveTab: (tab) => set({ activeTab: tab }),
    }), { name: 'UiStore' })
);