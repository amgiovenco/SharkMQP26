import shark1 from "../assets/photos/shark1.jpg";
import shark2 from "../assets/photos/shark2.jpg";
import FadeContent from "../components/FadeContent";
import GradientText from "../components/GradientText";

const HomePage = () => {
  return (
    <div className="relative w-screen h-screen flex items-center justify-center overflow-hidden">
      {/* <FadeContent blur={true} duration={1000} easing="ease-out" initialOpacity={0}> */}
        {/* Background image 1 */}
        <div
          className="absolute inset-0 bg-cover bg-center"
          style={{ backgroundImage: `url(${shark1})` }}
        />
      {/* </FadeContent> */}

      {/* Background image 2
      <FadeContent blur={true} duration={1000} easing="ease-out" initialOpacity={0}> */}
        {/* <div
          className="absolute inset-0 bg-cover bg-center"
          style={{ backgroundImage: `url(${shark2})` }}
        /> */}
      {/* </FadeContent> */}

      {/* White-to-transparent gradient at the top */}
      <div className="absolute inset-0 bg-gradient-to-b from-white via-white/70 to-transparent" />

      {/* Text overlay */}
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
      </div>
    </div>
  );
};

export default HomePage;
