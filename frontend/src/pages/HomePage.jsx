import shark1 from "../assets/photos/shark1.jpg";

const HomePage = () => {
  return (
    <div className="relative w-screen h-screen flex items-center justify-center overflow-hidden">
      {/* Background image */}
      <div
        className="absolute inset-0 bg-cover bg-center"
        style={{ backgroundImage: `url(${shark1})` }}
      />

      {/* White-to-transparent gradient at the top */}
      <div className="absolute inset-0 bg-gradient-to-b from-white via-white/70 to-transparent" />

      {/* Text overlay */}
      <div className="relative text-center px-4">
        <h1 className="text-4xl sm:text-5xl md:text-6xl font-extrabold">
          <span className="text-black font-nunitoSans">Where algorithms meet </span>
          <br />
          <span className="text-indigo-800 font-nunitoSans">oceans.</span>
        </h1>
      </div>
    </div>
  );
};

export default HomePage;
