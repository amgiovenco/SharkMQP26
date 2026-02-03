/**
 * usePermissions hook
 * Provides permission checks based on organization role
 */

import { useAuthStore } from '../stores/authStore';

export function usePermissions() {
    const { currentOrgRole, isSystemAdmin } = useAuthStore();

    const roleValue = currentOrgRole;

    return {
        // Case permissions
        canCreateCase: ['owner', 'admin', 'researcher'].includes(roleValue),
        canDeleteCase: ['owner', 'admin'].includes(roleValue),
        canEditCase: ['owner', 'admin', 'researcher'].includes(roleValue),

        // Job permissions
        canCreateJob: ['owner', 'admin', 'researcher', 'member'].includes(roleValue),
        canDeleteJob: ['owner', 'admin'].includes(roleValue),
        canViewAllJobs: ['owner', 'admin', 'researcher'].includes(roleValue),

        // Team management
        canManageTeam: ['owner', 'admin'].includes(roleValue),
        canManageCodes: ['owner', 'admin'].includes(roleValue),
        canChangeMemberRoles: ['owner', 'admin'].includes(roleValue),
        canRemoveMembers: ['owner', 'admin'].includes(roleValue),

        // Role checks
        isOwner: roleValue === 'owner',
        isAdmin: ['owner', 'admin'].includes(roleValue),
        isResearcher: ['owner', 'admin', 'researcher'].includes(roleValue),
        isMember: roleValue === 'member',
        isSystemAdmin: isSystemAdmin,

        // Current role
        currentRole: roleValue,
    };
}
