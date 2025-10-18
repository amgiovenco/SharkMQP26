import { createContext, useContext, useEffect, useState } from 'react';
import { io } from 'socket.io-client';

const SocketContext = createContext(null);

export const useSocket = () => useContext(SocketContext);

export const SocketProvider = ({ children }) => {
    const [socket, setSocket] = useState(null);
    
    useEffect(() => {
        // We only want to connect if the user is logged in, which we check with the JWT token
        const token = localStorage.getItem("jwt");
        if (token) {
            // Authenticate the websocket
            const newSocket = io({auth: { token }});

            // Logging for websocket
            newSocket.on('connect', () => {
                console.log('%cSocket connected!', 'color: green; font-weight: bold;', newSocket.id);
            });

            newSocket.on('disconnect', (reason) => {
                console.warn('Socket disconnected:', reason);
            });

            newSocket.on('connect_error', (err) => {
                console.error('Socket connection error:', err.message);
            });

            // Store the socket instance
            setSocket(newSocket);
    
            // Cleanup
            return () => newSocket.disconnect();
        }
    }, []);

    return (
        <SocketContext.Provider value={socket}>
            {children}
        </SocketContext.Provider>
    );
};