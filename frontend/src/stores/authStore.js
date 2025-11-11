// src/stores/authStore.js
import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';

export const useAuthStore = create(
    devtools(
        persist(
            (set) => ({
                jwt: null,
                userId: null,
                username: null,
                role: null,
                first_name: null,
                last_name: null,
                job_title: null,
                isAuthenticated: false,

                // Set authentication details
                setAuth: (jwt, userId, username, role, first_name, last_name, job_title) =>
                    set({
                        jwt,
                        userId,
                        username,
                        role,
                        first_name,
                        last_name,
                        job_title,
                        isAuthenticated: true,
                    }),

                // Clear authentication details
                clearAuth: () =>
                    set({
                        jwt: null,
                        userId: null,
                        username: null,
                        role: null,
                        first_name: null,
                        last_name: null,
                        job_title: null,
                        isAuthenticated: false,
                    }),

                // Set JWT token
                setJwt: (jwt) => set({ jwt }),

                // Update profile details
                updateProfile: (first_name, last_name, job_title) =>
                    set({
                        first_name,
                        last_name,
                        job_title,
                    }),
            }),
            {
                name: 'auth-storage',
                partialize: (state) => ({
                    jwt: state.jwt,
                }),
            }
        ),
        { name: 'AuthStore' }
    )
);