import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';

export const useAuthStore = create(
    devtools(
        persist(
            (set, get) => ({
                jwt: null,
                userId: null,
                username: null,
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
                setAuth: (jwt, userId, username, role, first_name, last_name, job_title, isSystemAdmin = false, organizations = []) =>
                    set({
                        jwt,
                        userId,
                        username,
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
                        username: null,
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
                }),
            }
        ),
        { name: 'AuthStore' }
    )
);