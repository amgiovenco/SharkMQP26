import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';
import { validateJwt } from '../utility/JWTUtil';

export const useAuthStore = create(
    devtools(
        persist(
            (set) => ({
                jwt: null,
                isAuthenticated: false,

                setAuth: (jwt) =>
                    set({
                        jwt,
                        isAuthenticated: true,
                    }),

                clearAuth: () =>
                    set({
                        jwt: null,
                        isAuthenticated: false,
                    }),
            }),
            {
                name: 'auth-storage',
                partialize: (state) => ({
                    jwt: state.jwt,
                    isAuthenticated: state.isAuthenticated,
                }),
                onRehydrateStorage: () => (state) => {
                    if (state?.jwt && !validateJwt(state.jwt)) {
                        state.clearAuth();
                    }
                },
            }
        ),
        { name: 'AuthStore' }
    )
);
