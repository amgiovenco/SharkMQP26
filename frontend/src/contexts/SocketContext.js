import { createContext, useContext, useEffect, useState } from 'react';
import { io } from 'socket.io-client';
import { useAuthStore } from '../stores/authStore';

const SocketContext = createContext(null);

export const useSocket = () => useContext(SocketContext);

export const SocketProvider = ({ children }) => {
    const [socket, setSocket] = useState(null);
    const { jwt, isAuthenticated } = useAuthStore();

    useEffect(() => {
        // Only connect if authenticated and have a token
        if (!isAuthenticated || !jwt) {
            // Clean up socket if logging out
            if (socket) {
                socket.disconnect();
                setSocket(null);
            }
            return;
        }

        // Create new socket connection with JWT auth
        const newSocket = io({
            auth: { token: jwt },
            reconnection: true,
            reconnectionDelay: 1000,
            reconnectionDelayMax: 5000,
            reconnectionAttempts: 5,
        });

        // Connection events
        newSocket.on('connect', () => {
            console.log('%cSocket connected!', 'color: green; font-weight: bold;', newSocket.id);
        });

        newSocket.on('disconnect', (reason) => {
            console.warn('Socket disconnected:', reason);
        });

        newSocket.on('connect_error', (err) => {
            console.error('Socket connection error:', err.message);
        });

        setSocket(newSocket);

        // Cleanup on unmount or when jwt changes
        return () => newSocket.disconnect();
    // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [jwt, isAuthenticated]); 

    return (
        <SocketContext.Provider value={socket}>
            {children}
        </SocketContext.Provider>
    );
};