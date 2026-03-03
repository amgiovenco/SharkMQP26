// import shark1 from "../assets/photos/shark1.jpg";
// import GradientText from "../components/GradientText";
// import FloatingLines from '../components/FloatingLines';
import { UnderwaterBackground } from '../components/underwater.jsx';
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
    <>
      <UnderwaterBackground>
        <div style={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          height: '100vh',
          width: '100vw',
        }}> 
        {/* <FloatingLines
          linesGradient={["#2681f7","#001670","#6f7076"]}
          animationSpeed={1}
          interactive={false}
          bendRadius={1}
          bendStrength={14.5}
          mouseDamping={0.07}
          parallax
          parallaxStrength={0.2}
        /> */}
      {/* Text */}
      <div style={{
        position: 'absolute',
        top: 0,
        left: 0,
        width: '100%',
        height: '100%',
        zIndex: 1,
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'flex-end',
        padding: '2rem 2rem 4rem 2rem'
      }}> 
      <div className="text-start text-white font-thin font-raleway" style={{
          fontSize: '6vw',
          letterSpacing: '0',
          lineHeight: '1.4'
        }}>
          WHERE&nbsp;&nbsp;&nbsp;&nbsp;ALGORITHMS
        </div>
        <div className="text-end text-white font-thin font-raleway" style={{
          fontSize: '6vw',
          letterSpacing: '0',
          lineHeight: '1.4',
          paddingLeft: '15vw'
        }}>
          MEET&nbsp;&nbsp;&nbsp;&nbsp;OCEANS
          </div>
      </div>
        </div>
      </UnderwaterBackground>
      </>
  );
};

export default HomePage;

