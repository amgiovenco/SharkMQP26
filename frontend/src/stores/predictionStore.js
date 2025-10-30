import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';

// Prediction History Store
export const usePredictionStore = create(
    devtools(
        persist(
            (set, get) => ({
                predictions: [],
                selectedPrediction: null,

                // Add a new prediction to history
                addPrediction: (prediction) =>
                    set((state) => ({
                        predictions: [prediction, ...state.predictions], // newest first
                    })),

                // Remove a prediction by jobId
                removePrediction: (jobId) =>
                    set((state) => ({
                        predictions: state.predictions.filter((p) => p.jobId !== jobId),
                        selectedPrediction:
                        state.selectedPrediction?.jobId === jobId
                            ? null
                            : state.selectedPrediction,
                    })),

                // Set the selected prediction
                selectPrediction: (prediction) =>
                    set({ selectedPrediction: prediction }),

                // Clear all history
                clearHistory: () =>
                    set({ predictions: [], selectedPrediction: null }),

                // Get prediction by jobId
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

        // Start a new prediction
        startPrediction: (fileName, jobId) =>
            set({
                isLoading: true,
                fileName,
                jobId,
                error: null,
                progress: 0,
            }),

        // Update progress
        setProgress: (progress) => set({ progress }),

        // Set error state
        setError: (error) =>
            set({
                error,
                isLoading: false,
            }),

        // Mark prediction as complete
        completePrediction: () =>
            set({
                isLoading: false,
                progress: 100,
            }),

        // Reset current prediction state
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