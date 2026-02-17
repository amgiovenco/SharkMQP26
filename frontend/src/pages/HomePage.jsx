// import shark1 from "../assets/photos/shark1.jpg";
// import GradientText from "../components/GradientText";
import FloatingLines from '../components/FloatingLines';


const HomePage = () => {
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
      }}>
        <span className="text-white font-raleway" style={{ fontSize: '3rem' }}>
          Where algorithms meet oceans
        </span>
      </div>
    </div>
  );
};

export default HomePage;

