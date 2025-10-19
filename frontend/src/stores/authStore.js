import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';

// Auth Store
export const useAuthStore = create(
    devtools(
        persist(
            (set) => ({
                jwt: null,
                userId: null,
                username: null,
                isAuthenticated: false,

                setAuth: (jwt, userId, username) =>
                    set({
                        jwt,
                        userId,
                        username,
                        isAuthenticated: true,
                    }),

                clearAuth: () =>
                    set({
                        jwt: null,
                        userId: null,
                        username: null,
                        isAuthenticated: false,
                    }),

                setJwt: (jwt) => set({ jwt }),
            }),
            {
                name: 'auth-storage', // localStorage key
                partialize: (state) => ({
                    jwt: state.jwt,
                    userId: state.userId,
                    username: state.username,
                    isAuthenticated: state.isAuthenticated,
                }),
            }
        ),
        { name: 'AuthStore' }
    )
);
