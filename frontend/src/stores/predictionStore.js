import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';

// Prediction History Store
export const usePredictionStore = create(
    devtools(
        persist(
            (set, get) => ({
                predictions: [],
                selectedPrediction: null,

                addPrediction: (prediction) =>
                    set((state) => ({
                        predictions: [prediction, ...state.predictions], // newest first
                    })),

                removePrediction: (jobId) =>
                    set((state) => ({
                        predictions: state.predictions.filter((p) => p.jobId !== jobId),
                        selectedPrediction:
                        state.selectedPrediction?.jobId === jobId
                            ? null
                            : state.selectedPrediction,
                    })),

                selectPrediction: (prediction) =>
                    set({ selectedPrediction: prediction }),

                clearHistory: () =>
                    set({ predictions: [], selectedPrediction: null }),

                getPredictionByJobId: (jobId) => {
                    const { predictions } = get();
                    return predictions.find((p) => p.jobId === jobId);
                },
            }),
            {
                name: 'prediction-storage',
                partialize: (state) => ({
                    predictions: state.predictions,
                }),
            }
        ),
        { name: 'PredictionStore' }
    )
);

// Current Prediction (In-Flight) Store
export const useCurrentPredictionStore = create(
    devtools((set) => ({
        isLoading: false,
        jobId: null,
        fileName: null,
        error: null,
        progress: 0,

        startPrediction: (fileName, jobId) =>
            set({
                isLoading: true,
                fileName,
                jobId,
                error: null,
                progress: 0,
            }),

        setProgress: (progress) => set({ progress }),

        setError: (error) =>
            set({
                error,
                isLoading: false,
            }),

        completePrediction: () =>
            set({
                isLoading: false,
                progress: 100,
            }),

        resetPrediction: () =>
            set({
                isLoading: false,
                jobId: null,
                fileName: null,
                error: null,
                progress: 0,
            }),
    }), { name: 'CurrentPredictionStore' })
);