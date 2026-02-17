// import shark1 from "../assets/photos/shark1.jpg";
// import GradientText from "../components/GradientText";
import FloatingLines from '../components/FloatingLines';
import { useEffect } from 'react';


const HomePage = () => {
  useEffect(() => {
    // Disable scroll on mount
    document.body.style.overflow = 'hidden';
    
    // Re-enable scroll on unmount
    return () => {
      document.body.style.overflow = 'auto';
    };
  }, []);

  return (
    <div style={{ 
      width: '100vw', 
      height: '100vh', 
      position: 'relative',
      overflow: 'hidden' 
    }}>

      {/* Background - fills entire container */}
      <div style={{
        position: 'absolute',
        top: 0,
        left: 0,
        width: '100%',
        height: '100%',
        zIndex: 0
      }}>
        <FloatingLines
          linesGradient={["#2681f7","#001670","#6f7076"]}
          animationSpeed={1}
          interactive={false}
          bendRadius={1}
          bendStrength={14.5}
          mouseDamping={0.07}
          parallax
          parallaxStrength={0.2}
        />
      </div>

      {/* Text - sits on top */}
      {/* Text */}
      <div style={{
        position: 'absolute',
        top: 0,
        left: 0,
        width: '100%',
        height: '100%',
        zIndex: 1,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '0 2rem'
      }}>
        <h1 className="font-raleway text-white" style={{ 
          fontSize: '5rem',
          fontWeight: 100,
          textAlign: 'center',
          lineHeight: '1.2',
          letterSpacing: '0.15em'
        }}>
          WHERE ALGORITHMS
          <br />
          MEET OCEANS
        </h1>
      </div>
    </div>
  );
};

export default HomePage;

