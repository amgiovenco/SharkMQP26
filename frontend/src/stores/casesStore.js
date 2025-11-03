import { create } from 'zustand';
import { devtools } from 'zustand/middleware';

export const useCasesStore = create(
    devtools(
        (set, get) => ({
                cases: [],
                isLoading: false,
                error: null,
                page: 1,
                perPage: 20,
                total: 0,

                // Set all cases (used after login)
                setCases: (cases) =>
                    set({
                        cases,
                    }),

                // Add a single case (used when creating new case)
                addCase: (newCase) =>
                    set((state) => ({
                        cases: [newCase, ...state.cases],
                    })),

                // Remove a case
                removeCase: (caseId) =>
                    set((state) => ({
                        cases: state.cases.filter((c) => c.id !== caseId),
                    })),

                // Update a case
                updateCase: (caseId, updatedData) =>
                    set((state) => ({
                        cases: state.cases.map((c) =>
                        c.id === caseId ? { ...c, ...updatedData } : c
                        ),
                    })),

                // Pagination helpers
                setPage: (page) => set({ page }),
                setPerPage: (perPage) => set({ perPage }),
                setTotal: (total) => set({ total }),

                // Loading/error states
                setIsLoading: (isLoading) => set({ isLoading }),
                setError: (error) => set({ error }),

                // Clear all cases
                clearCases: () =>
                    set({
                        cases: [],
                        page: 1,
                        total: 0,
                        error: null,
                    }),

                // Get case by ID
                getCaseById: (caseId) => {
                    const { cases } = get();
                    return cases.find((c) => c.id === caseId);
                },
            }),
        { name: 'CasesStore' }
    )
);