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
      {/* Background */}
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
    </div>
  );
};

export default HomePage;

{/*Text overlay
      <div className="relative text-center px-4">
        <h1 className="text-4xl sm:text-5xl md:text-6xl font-extrabold">
          <span className="text-black font-nunitoSans">
            Where algorithms meet{" "}
          </span>
          <br />
          <GradientText
            colors={["#252491", "#3B4FFF", "#0CB6FF", "#44B3D3", "#9FB6C4"]}
            animationSpeed={11}
            showBorder={false}
            className="custom-class"
          >
            oceans
          </GradientText>
        </h1>
      </div> */ }