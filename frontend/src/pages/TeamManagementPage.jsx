/**
 * TeamManagementPage
 * Manage organization members and registration codes
 */

import { useState, useEffect, useCallback } from 'react';
import toast from 'react-hot-toast';
import { useAuthStore } from '../stores/authStore';
import { usePermissions } from '../hooks/usePermissions';
import { apiFetch } from '../utility/ApiFetch';

export function TeamManagementPage() {
    const { currentOrganization } = useAuthStore();
    const { canManageTeam, canManageCodes } = usePermissions();

    const [members, setMembers] = useState([]);
    const [codes, setCodes] = useState([]);
    const [showCreateCode, setShowCreateCode] = useState(false);
    const [selectedCodeRole, setSelectedCodeRole] = useState('researcher');
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const loadMembers = useCallback(async () => {
        try {
            setLoading(true);
            const data = await apiFetch(`/organizations/${currentOrganization.id}/members`);
            setMembers(data.members || []);
            setError(null);
        } catch (err) {
            setError(err.message);
            console.error('Error loading members:', err);
        } finally {
            setLoading(false);
        }
    }, [currentOrganization]);

    const loadCodes = useCallback(async () => {
        try {
            const data = await apiFetch(`/organizations/${currentOrganization.id}/codes`);
            setCodes(data.codes || []);
        } catch (err) {
            console.error('Error loading codes:', err);
        }
    }, [currentOrganization]);

    useEffect(() => {
        loadMembers();
        if (canManageCodes) {
            loadCodes();
        }
    }, [loadMembers, loadCodes, canManageCodes]);

    const createCode = async () => {
        try {
            const data = await apiFetch(`/organizations/${currentOrganization.id}/codes`, {
                method: 'POST',
                body: JSON.stringify({ role: selectedCodeRole }),
            });
            toast.success(`Registration code created: ${data.code}`);
            loadCodes();
            setShowCreateCode(false);
            setSelectedCodeRole('researcher');
        } catch (err) {
            toast.error(`Error: ${err.message}`);
            console.error('Error creating code:', err);
        }
    };

    const removeMember = async (userId) => {
        if (!window.confirm('Remove this member?')) return;

        try {
            await apiFetch(`/organizations/${currentOrganization.id}/members/${userId}`, {
                method: 'DELETE',
            });
            toast.success('Member removed successfully');
            loadMembers();
        } catch (err) {
            toast.error(`Error: ${err.message}`);
        }
    };

    const changeRole = async (userId, newRole) => {
        try {
            await apiFetch(`/organizations/${currentOrganization.id}/members/${userId}`, {
                method: 'PATCH',
                body: JSON.stringify({ new_role: newRole }),
            });
            toast.success('Role updated successfully');
            loadMembers();
        } catch (err) {
            toast.error(`Error: ${err.message}`);
        }
    };

    const disableCode = async (codeId) => {
        if (!window.confirm('Disable this registration code?')) return;

        try {
            await apiFetch(`/organizations/${currentOrganization.id}/codes/${codeId}`, {
                method: 'DELETE',
            });
            toast.success('Registration code disabled');
            loadCodes();
        } catch (err) {
            toast.error(`Error: ${err.message}`);
        }
    };

    if (!canManageTeam) {
        return (
            <div className="flex items-center justify-center min-h-screen">
                <div className="text-center">
                    <h1 className="text-2xl font-bold text-gray-900 mb-2">
                        Access Denied
                    </h1>
                    <p className="text-gray-600">
                        Only owners and admins can manage team members.
                    </p>
                </div>
            </div>
        );
    }

    return (
        <div className="min-h-screen bg-gray-50 py-8">
            <div className="max-w-6xl mx-auto px-4">
                <h1 className="text-3xl font-bold text-gray-900 mb-8">
                    Team Management
                </h1>

                {error && (
                    <div className="mb-6 p-4 bg-red-100 text-red-700 rounded">
                        {error}
                    </div>
                )}

                {/* Registration Codes Section */}
                {canManageCodes && (
                    <div className="mb-8">
                        <div className="flex justify-between items-center mb-4">
                            <h2 className="text-2xl font-bold text-gray-900 text-gray-900">
                                Registration Codes
                            </h2>
                            <button
                                onClick={() => setShowCreateCode(true)}
                                className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 transition-colors"
                            >
                                Create Code
                            </button>
                        </div>

                        {showCreateCode && (
                            <div className="mb-4 p-4 bg-gray-100 bg-white rounded">
                                <p className="mb-4 font-medium text-gray-900 text-gray-900">
                                    Select role for new code:
                                </p>
                                <div className="flex gap-2 mb-4">
                                    {['admin', 'researcher', 'member'].map((role) => (
                                        <button
                                            key={role}
                                            onClick={() => setSelectedCodeRole(role)}
                                            className={`px-3 py-2 rounded capitalize transition-colors ${
                                                selectedCodeRole === role
                                                    ? 'bg-blue-600 text-white'
                                                    : 'bg-gray-300 bg-gray-300 text-gray-900 text-gray-900 hover:bg-gray-400 hover:bg-gray-400'
                                            }`}
                                        >
                                            {role}
                                        </button>
                                    ))}
                                </div>
                                <div className="flex gap-2">
                                    <button
                                        onClick={createCode}
                                        className="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700 transition-colors"
                                    >
                                        Create
                                    </button>
                                    <button
                                        onClick={() => setShowCreateCode(false)}
                                        className="px-4 py-2 bg-gray-400 text-white rounded hover:bg-gray-500 transition-colors"
                                    >
                                        Cancel
                                    </button>
                                </div>
                            </div>
                        )}

                        <div className="bg-white bg-white shadow rounded-lg overflow-hidden">
                            <table className="w-full">
                                <thead className="bg-gray-50 bg-gray-50">
                                    <tr>
                                        <th className="px-6 py-3 text-left text-sm font-semibold text-gray-900 text-gray-900">
                                            Code
                                        </th>
                                        <th className="px-6 py-3 text-left text-sm font-semibold text-gray-900 text-gray-900">
                                            Role
                                        </th>
                                        <th className="px-6 py-3 text-left text-sm font-semibold text-gray-900 text-gray-900">
                                            Uses
                                        </th>
                                        <th className="px-6 py-3 text-left text-sm font-semibold text-gray-900 text-gray-900">
                                            Status
                                        </th>
                                        <th className="px-6 py-3 text-right text-sm font-semibold text-gray-900 text-gray-900">
                                            Actions
                                        </th>
                                    </tr>
                                </thead>
                                <tbody className="divide-y divide-gray-200 divide-gray-200">
                                    {codes.length === 0 ? (
                                        <tr>
                                            <td
                                                colSpan="5"
                                                className="px-6 py-4 text-center text-gray-500 text-gray-600"
                                            >
                                                No registration codes yet
                                            </td>
                                        </tr>
                                    ) : (
                                        codes.map((code) => (
                                            <tr
                                                key={code.id}
                                                className="hover:bg-gray-50 hover:bg-gray-50 transition-colors"
                                            >
                                                <td className="px-6 py-4 font-mono text-gray-900 text-gray-900">
                                                    {code.code}
                                                </td>
                                                <td className="px-6 py-4 capitalize text-gray-900 text-gray-900">
                                                    {code.role}
                                                </td>
                                                <td className="px-6 py-4 text-gray-600 text-gray-600">
                                                    {code.times_used}
                                                    {code.uses_remaining
                                                        ? ` / ${code.uses_remaining + code.times_used}`
                                                        : ' / ∞'}
                                                </td>
                                                <td className="px-6 py-4 text-gray-900 text-gray-900">
                                                    <span
                                                        className={`px-2 py-1 rounded text-sm ${
                                                            code.status === 'active'
                                                                ? 'bg-green-100 text-green-800 bg-green-100 text-green-800'
                                                                : 'bg-red-100 text-red-800 bg-red-100 text-red-800'
                                                        }`}
                                                    >
                                                        {code.status}
                                                    </span>
                                                </td>
                                                <td className="px-6 py-4 text-right">
                                                    {code.status === 'active' && (
                                                        <button
                                                            onClick={() => disableCode(code.id)}
                                                            className="text-red-600 hover:text-red-800 text-red-600 hover:text-red-800 transition-colors"
                                                        >
                                                            Disable
                                                        </button>
                                                    )}
                                                </td>
                                            </tr>
                                        ))
                                    )}
                                </tbody>
                            </table>
                        </div>
                    </div>
                )}

                {/* Members Section */}
                <div>
                    <h2 className="text-2xl font-bold text-gray-900 text-gray-900 mb-4">
                        Team Members ({members.length})
                    </h2>

                    <div className="bg-white bg-white shadow rounded-lg overflow-hidden">
                        <table className="w-full">
                            <thead className="bg-gray-50 bg-gray-50">
                                <tr>
                                    <th className="px-6 py-3 text-left text-sm font-semibold text-gray-900 text-gray-900">
                                        Username
                                    </th>
                                    <th className="px-6 py-3 text-left text-sm font-semibold text-gray-900 text-gray-900">
                                        Name
                                    </th>
                                    <th className="px-6 py-3 text-left text-sm font-semibold text-gray-900 text-gray-900">
                                        Role
                                    </th>
                                    <th className="px-6 py-3 text-left text-sm font-semibold text-gray-900 text-gray-900">
                                        Joined
                                    </th>
                                    <th className="px-6 py-3 text-right text-sm font-semibold text-gray-900 text-gray-900">
                                        Actions
                                    </th>
                                </tr>
                            </thead>
                            <tbody className="divide-y divide-gray-200 divide-gray-200">
                                {loading ? (
                                    <tr>
                                        <td colSpan="5" className="px-6 py-4 text-center">
                                            Loading...
                                        </td>
                                    </tr>
                                ) : members.length === 0 ? (
                                    <tr>
                                        <td
                                            colSpan="5"
                                            className="px-6 py-4 text-center text-gray-500 text-gray-600"
                                        >
                                            No team members
                                        </td>
                                    </tr>
                                ) : (
                                    members.map((member) => (
                                        <tr
                                            key={member.user_id}
                                            className="hover:bg-gray-50 hover:bg-gray-50 transition-colors"
                                        >
                                            <td className="px-6 py-4 text-gray-900 text-gray-900 font-medium">
                                                {member.username}
                                            </td>
                                            <td className="px-6 py-4 text-gray-600 text-gray-600">
                                                {member.full_name || '-'}
                                            </td>
                                            <td className="px-6 py-4">
                                                <select
                                                    value={member.role}
                                                    onChange={(e) =>
                                                        changeRole(member.user_id, e.target.value)
                                                    }
                                                    className="px-2 py-1 border border-gray-300 border-gray-300 rounded bg-white bg-gray-50 text-gray-900 text-gray-900"
                                                >
                                                    <option value="owner">Owner</option>
                                                    <option value="admin">Admin</option>
                                                    <option value="researcher">Researcher</option>
                                                    <option value="member">Member</option>
                                                </select>
                                            </td>
                                            <td className="px-6 py-4 text-gray-600 text-gray-600">
                                                {new Date(member.joined_at).toLocaleDateString()}
                                            </td>
                                            <td className="px-6 py-4 text-right">
                                                <button
                                                    onClick={() => removeMember(member.user_id)}
                                                    className="text-red-600 hover:text-red-800 text-red-600 hover:text-red-800 transition-colors"
                                                >
                                                    Remove
                                                </button>
                                            </td>
                                        </tr>
                                    ))
                                )}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>
    );
}
