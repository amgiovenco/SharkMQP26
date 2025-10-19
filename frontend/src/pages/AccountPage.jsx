// view account details and edit them
import { useState } from 'react';
import { useAuthStore } from '../stores/authStore';
import { apiFetch } from '../utility/ApiFetch';

const AccountPage = () => {
    const auth = useAuthStore();
    const [form, setForm] = useState({
        first_name: auth.first_name || '',
        last_name: auth.last_name || '',
        job_title: auth.job_title || '',
        password: '',
        password_confirm: '',
    });
    const [message, setMessage] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [showPasswordFields, setShowPasswordFields] = useState(false);

    const handleInputChange = (e) => {
        const { name, value } = e.target;
        setForm((prev) => ({ ...prev, [name]: value }));
    };

    const handleUpdateProfile = async (e) => {
        e.preventDefault();
        setIsLoading(true);
        setMessage('');

        try {
            const updateData = {
                first_name: form.first_name,
                last_name: form.last_name,
                job_title: form.job_title,
            };

            await apiFetch('/auth/users/profile', {
                method: 'PUT',
                body: JSON.stringify(updateData),
            });

            setMessage('Profile updated successfully');

            auth.updateProfile(form.first_name, form.last_name, form.job_title);
            
            setForm((prev) => ({
                ...prev,
                first_name: form.first_name,
                last_name: form.last_name,
                job_title: form.job_title,
            }));
        } catch (error) {
            setMessage(`Error: ${error.message}`);
        } finally {
            setIsLoading(false);
        }
    };

    const handleChangePassword = async (e) => {
        e.preventDefault();

        if (form.password !== form.password_confirm) {
            setMessage('Passwords do not match');
            return;
        }

        if (form.password.length < 8) {
            setMessage('Password must be at least 8 characters');
            return;
        }

        setIsLoading(true);
        setMessage('');

        try {
            await apiFetch('/auth/users/password', {
                method: 'PUT',
                body: JSON.stringify({ new_password: form.password }),
            });

            setMessage('Password changed successfully');
            setForm((prev) => ({ ...prev, password: '', password_confirm: '' }));
            setShowPasswordFields(false);
        } catch (error) {
            setMessage(`Error: ${error.message}`);
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="p-8 max-w-2xl">
            <h1 className="text-3xl font-bold mb-8">Account Settings</h1>

            {/* Account Info */}
            <div className="mb-8 p-4 border rounded">
                <h2 className="text-xl font-semibold mb-4">Account Information</h2>
                <p className="mb-2">
                    <strong>Username:</strong> {auth.username}
                </p>
                <p className="mb-2">
                    <strong>Role:</strong> {auth.role}
                </p>
            </div>

            {/* Update Profile */}
            <div className="mb-8 p-4 border rounded">
                <h2 className="text-xl font-semibold mb-4">Update Profile</h2>
                <form onSubmit={handleUpdateProfile} className="space-y-4">
                    <div>
                        <label className="block text-sm font-medium mb-1">First Name</label>
                        <input
                            type="text"
                            name="first_name"
                            value={form.first_name}
                            onChange={handleInputChange}
                            className="w-full p-2 border rounded"
                            placeholder="Enter first name"
                        />
                    </div>

                    <div>
                        <label className="block text-sm font-medium mb-1">Last Name</label>
                        <input
                            type="text"
                            name="last_name"
                            value={form.last_name}
                            onChange={handleInputChange}
                            className="w-full p-2 border rounded"
                            placeholder="Enter last name"
                        />
                    </div>

                    <div>
                        <label className="block text-sm font-medium mb-1">Job Title</label>
                        <input
                            type="text"
                            name="job_title"
                            value={form.job_title}
                            onChange={handleInputChange}
                            className="w-full p-2 border rounded"
                            placeholder="Enter job title"
                        />
                    </div>

                    <button
                        type="submit"
                        disabled={isLoading}
                        className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50"
                    >
                        {isLoading ? 'Saving...' : 'Update Profile'}
                    </button>
                </form>
            </div>

            {/* Change Password */}
            <div className="mb-8 p-4 border rounded">
                <h2 className="text-xl font-semibold mb-4">Change Password</h2>
                
                {!showPasswordFields ? (
                    <button
                        onClick={() => setShowPasswordFields(true)}
                        className="px-4 py-2 bg-gray-600 text-white rounded hover:bg-gray-700"
                    >
                        Change Password
                    </button>
                ) : (
                    <form onSubmit={handleChangePassword} className="space-y-4">
                        <div>
                            <label className="block text-sm font-medium mb-1">New Password</label>
                            <input
                                type="password"
                                name="password"
                                value={form.password}
                                onChange={handleInputChange}
                                className="w-full p-2 border rounded"
                                placeholder="Enter new password (min 8 characters)"
                                required
                            />
                        </div>

                        <div>
                            <label className="block text-sm font-medium mb-1">Confirm Password</label>
                            <input
                                type="password"
                                name="password_confirm"
                                value={form.password_confirm}
                                onChange={handleInputChange}
                                className="w-full p-2 border rounded"
                                placeholder="Confirm new password"
                                required
                            />
                        </div>

                        <div className="space-x-2">
                            <button
                                type="submit"
                                disabled={isLoading}
                                className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50"
                            >
                                {isLoading ? 'Updating...' : 'Update Password'}
                            </button>
                            <button
                                type="button"
                                onClick={() => setShowPasswordFields(false)}
                                className="px-4 py-2 bg-gray-400 text-white rounded hover:bg-gray-500"
                            >
                                Cancel
                            </button>
                        </div>
                    </form>
                )}
            </div>

            {/* Messages */}
            {message && (
                <div className={`p-4 rounded ${message.includes('Error') ? 'bg-red-100 text-red-800' : 'bg-green-100 text-green-800'}`}>
                    {message}
                </div>
            )}
        </div>
    );
};

export default AccountPage;