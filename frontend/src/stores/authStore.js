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

        setJwt: (jwt) => set({ jwt }),

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
          userId: state.userId,
          username: state.username,
          role: state.role,
          first_name: state.first_name,
          last_name: state.last_name,
          job_title: state.job_title,
          isAuthenticated: state.isAuthenticated,
        }),
      }
    ),
    { name: 'AuthStore' }
  )
);