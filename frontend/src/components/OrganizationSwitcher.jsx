/**
 * OrganizationSwitcher component
 * Allows users to switch between organizations they belong to
 */

import { useState } from 'react';
import { useAuthStore } from '../stores/authStore';

export function OrganizationSwitcher() {
    const { organizations, currentOrganization, switchOrganization } = useAuthStore();
    const [isOpen, setIsOpen] = useState(false);

    // Don't show switcher if only one organization
    if (!organizations || organizations.length <= 1) {
        return null;
    }

    const handleSwitchOrg = (orgId) => {
        switchOrganization(orgId);
        setIsOpen(false);
    };

    return (
        <div className="relative">
            <button
                onClick={() => setIsOpen(!isOpen)}
                className="flex items-center gap-2 px-3 py-2 bg-gray-100 dark:bg-gray-700 rounded hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors"
                title="Switch organization"
            >
                <span className="text-sm font-medium">
                    {currentOrganization?.name || 'Select Organization'}
                </span>
                <svg
                    className={`w-4 h-4 transition-transform ${isOpen ? 'rotate-180' : ''}`}
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                >
                    <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M19 9l-7 7-7-7"
                    />
                </svg>
            </button>

            {isOpen && (
                <div className="absolute right-0 mt-2 w-64 bg-white dark:bg-gray-800 shadow-lg rounded border border-gray-200 dark:border-gray-700 z-50">
                    {organizations.map((org) => (
                        <button
                            key={org.id}
                            onClick={() => handleSwitchOrg(org.id)}
                            className={`w-full px-4 py-3 text-left hover:bg-gray-50 dark:hover:bg-gray-700 border-b border-gray-100 dark:border-gray-700 last:border-0 transition-colors ${
                                org.id === currentOrganization?.id
                                    ? 'bg-blue-50 dark:bg-blue-900'
                                    : ''
                            }`}
                        >
                            <div className="font-medium text-gray-900 dark:text-white">
                                {org.name}
                            </div>
                            <div className="text-sm text-gray-500 dark:text-gray-400 capitalize">
                                {org.role}
                            </div>
                        </button>
                    ))}
                </div>
            )}
        </div>
    );
}
