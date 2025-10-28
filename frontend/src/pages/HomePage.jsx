const HomePage = () => {
  return (
    <div className="w-full h-full flex flex-col items-center justify-center space-y-8 pt-20 pb-20">
      <div className="w-[683px] h-32 text-center">
        <span class="text-black text-6xl font-extrabold [text-shadow:_0px_4px_5px_rgb(0_0_0_/_0.20)]">
          Where algorithms meet{" "}
        </span>
        <span class="text-indigo-900 text-6xl font-extrabold font- [text-shadow:_0px_4px_5px_rgb(0_0_0_/_0.20)]">
          oceans
        </span>
        <span class="text-black text-6xl font-extrabold font- [text-shadow:_0px_4px_5px_rgb(0_0_0_/_0.20)]">
          .
        </span>
      </div>
      {/* rotation of photos i have to find others. maybe different gradient */}
      {/* <div className="w-[1920px] h-[1238px] relative">
        <div className="w-[1920px] h-[1238px] left-0 top-0 absolute bg-gradient-to-b from-black/0 via-black/25 to-black" />
        <img
          className="w-[2232.58px] h-[1255.82px] left-[-157.75px] top-[-74.39px] absolute"
          src="./assets/photos/sarkImage.png"
        />
      </div> */}
    </div>
  );
};

export default HomePage;
