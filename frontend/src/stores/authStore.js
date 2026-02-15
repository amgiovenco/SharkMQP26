import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';
import { validateJwt } from '../utility/JWTUtil';

export const useAuthStore = create(
    devtools(
        persist(
            (set, get) => ({
                jwt: null,
                userId: null,
                email: null,
                role: null,
                first_name: null,
                last_name: null,
                job_title: null,
                isSystemAdmin: false,
                isAuthenticated: false,

                // Multi-tenancy
                organizations: [],
                currentOrganization: null,
                currentOrgRole: null,

                // Set authentication details
                setAuth: (jwt, userId, email, role, first_name, last_name, job_title, isSystemAdmin = false, organizations = []) =>
                    set({
                        jwt,
                        userId,
                        email,
                        role,
                        first_name,
                        last_name,
                        job_title,
                        isSystemAdmin,
                        organizations,
                        currentOrganization: organizations.length > 0 ? organizations[0] : null,
                        currentOrgRole: organizations.length > 0 ? organizations[0].role : null,
                        isAuthenticated: true,
                    }),

                // Clear authentication details
                clearAuth: () =>
                    set({
                        jwt: null,
                        userId: null,
                        email: null,
                        role: null,
                        first_name: null,
                        last_name: null,
                        job_title: null,
                        isSystemAdmin: false,
                        organizations: [],
                        currentOrganization: null,
                        currentOrgRole: null,
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

                // Switch to a different organization
                switchOrganization: (orgId) => {
                    const { organizations } = get();
                    const org = organizations.find(o => o.id === orgId);
                    if (org) {
                        set({
                            currentOrganization: org,
                            currentOrgRole: org.role,
                        });
                        return true;
                    }
                    return false;
                },

                // Update organizations list (after user updates)
                setOrganizations: (organizations) =>
                    set({
                        organizations,
                        currentOrganization: organizations.length > 0 ? organizations[0] : null,
                        currentOrgRole: organizations.length > 0 ? organizations[0].role : null,
                    }),
            }),
            {
                name: 'auth-storage',
                partialize: (state) => ({
                    jwt: state.jwt,
                    isAuthenticated: state.isAuthenticated,
                    userId: state.userId,
                    email: state.email,
                    role: state.role,
                    first_name: state.first_name,
                    last_name: state.last_name,
                    job_title: state.job_title,
                    isSystemAdmin: state.isSystemAdmin,
                    organizations: state.organizations,
                    currentOrganization: state.currentOrganization,
                    currentOrgRole: state.currentOrgRole,
                }),
                onRehydrateStorage: () => (state) => {
                    // Validate JWT after rehydration from localStorage
                    if (state?.jwt && !validateJwt(state.jwt)) {
                        // JWT is expired or invalid, clear auth
                        state.clearAuth();
                    }
                },
            }
        ),
        { name: 'AuthStore' }
    )
);