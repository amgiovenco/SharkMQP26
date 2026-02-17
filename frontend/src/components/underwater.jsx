import { useEffect, useRef } from "react";

export function UnderwaterBackground({
  className = "",
  children,
  intensity = 1,
  speed = 1,
}) {
  const canvasRef = useRef(null);
  const containerRef = useRef(null);

  // Particles animation
  useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const rect = container.getBoundingClientRect();
    let width = rect.width;
    let height = rect.height;
    canvas.width = width;
    canvas.height = height;

    let animationId;
    let tick = 0;

    const particles = Array.from({ length: 40 }, () => ({
      x: Math.random() * width,
      y: Math.random() * height,
      size: 1 + Math.random() * 2,
      speed: 0.3 + Math.random() * 0.4,
      opacity: 0.4 + Math.random() * 0.4,
      wobbleOffset: Math.random() * Math.PI * 2,
    }));

    const handleResize = () => {
      const rect = container.getBoundingClientRect();
      width = rect.width;
      height = rect.height;
      canvas.width = width;
      canvas.height = height;
    };

    const ro = new ResizeObserver(handleResize);
    ro.observe(container);

    const animate = () => {
      tick += 0.02 * speed;
      ctx.clearRect(0, 0, width, height);

      for (const p of particles) {
        p.y -= p.speed * speed;
        p.x += Math.sin(tick * 1.5 + p.wobbleOffset) * 0.4;

        if (p.y < -10) {
          p.y = height + 10;
          p.x = Math.random() * width;
        }

        ctx.beginPath();
        ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(180, 230, 255, ${p.opacity})`;
        ctx.fill();
      }

      animationId = requestAnimationFrame(animate);
    };

    animationId = requestAnimationFrame(animate);

    return () => {
      cancelAnimationFrame(animationId);
      ro.disconnect();
    };
  }, [speed]);

  const duration1 = 8 / speed;
  const duration2 = 12 / speed;
  const duration3 = 10 / speed;

  return (
    <div
      ref={containerRef}
      className={`fixed inset-0 overflow-hidden ${className}`}
      style={{
        background: "linear-gradient(180deg, #006994 0%, #004466 40%, #002233 100%)",
      }}
    >
      {/* Caustic light layers */}
      <div className="absolute inset-0 overflow-hidden">
        <div
          className="absolute opacity-30"
          style={{
            inset: "-50%",
            background: `
              radial-gradient(ellipse 40% 30% at 30% 30%, rgba(100, 200, 255, ${0.4 * intensity}), transparent),
              radial-gradient(ellipse 35% 40% at 70% 40%, rgba(80, 180, 255, ${0.3 * intensity}), transparent),
              radial-gradient(ellipse 45% 35% at 50% 60%, rgba(120, 220, 255, ${0.35 * intensity}), transparent)
            `,
            animation: `caustic1 ${duration1}s ease-in-out infinite`,
            filter: "blur(40px)",
          }}
        />
        <div
          className="absolute opacity-25"
          style={{
            inset: "-50%",
            background: `
              radial-gradient(ellipse 50% 40% at 60% 35%, rgba(150, 230, 255, ${0.35 * intensity}), transparent),
              radial-gradient(ellipse 40% 45% at 25% 55%, rgba(100, 200, 255, ${0.3 * intensity}), transparent)
            `,
            animation: `caustic2 ${duration2}s ease-in-out infinite`,
            filter: "blur(50px)",
          }}
        />
        <div
          className="absolute opacity-20"
          style={{
            inset: "-50%",
            background: `
              radial-gradient(ellipse 35% 50% at 45% 45%, rgba(180, 240, 255, ${0.4 * intensity}), transparent),
              radial-gradient(ellipse 45% 35% at 75% 65%, rgba(130, 210, 255, ${0.3 * intensity}), transparent)
            `,
            animation: `caustic3 ${duration3}s ease-in-out infinite`,
            filter: "blur(35px)",
          }}
        />
      </div>

      {/* Light rays */}
      <div className="absolute inset-0 overflow-hidden">
        {[0, 1, 2, 3, 4].map(i => (
          <div
            key={i}
            className="absolute top-0"
            style={{
              left: `${15 + i * 18}%`,
              width: "8%",
              height: "100%",
              background: `linear-gradient(180deg, rgba(180, 230, 255, ${0.12 * intensity}) 0%, rgba(150, 210, 255, ${0.04 * intensity}) 50%, transparent 80%)`,
              transform: "skewX(-5deg)",
              animation: `ray ${6 + i * 2}s ease-in-out infinite`,
              animationDelay: `${i * -1.5}s`,
              filter: "blur(8px)",
            }}
          />
        ))}
      </div>

      {/* Particles canvas */}
      <canvas ref={canvasRef} className="absolute inset-0 h-full w-full" />

      {/* Surface shimmer */}
      <div
        className="pointer-events-none absolute inset-x-0 top-0 h-1/4"
        style={{
          background: `linear-gradient(180deg, rgba(150, 220, 255, ${0.2 * intensity}) 0%, transparent 100%)`,
        }}
      />

      {/* Depth fade */}
      <div
        className="pointer-events-none absolute inset-x-0 bottom-0 h-1/3"
        style={{
          background: "linear-gradient(0deg, rgba(0, 15, 30, 0.7) 0%, transparent 100%)",
        }}
      />

      {/* Vignette */}
      <div
        className="pointer-events-none absolute inset-0"
        style={{
          background:
            "radial-gradient(ellipse at 50% 30%, transparent 0%, transparent 50%, rgba(0, 20, 40, 0.6) 100%)",
        }}
      />

      {/* Content layer */}
      {children && <div className="relative z-10 h-full w-full">{children}</div>}

      <style>{`
        @keyframes caustic1 {
          0%, 100% { transform: translate(0%, 0%) scale(1); }
          33% { transform: translate(5%, 3%) scale(1.05); }
          66% { transform: translate(-3%, -2%) scale(0.95); }
        }
        @keyframes caustic2 {
          0%, 100% { transform: translate(0%, 0%) scale(1); }
          50% { transform: translate(-6%, 4%) scale(1.08); }
        }
        @keyframes caustic3 {
          0%, 100% { transform: translate(0%, 0%) scale(1.02); }
          33% { transform: translate(4%, -3%) scale(0.96); }
          66% { transform: translate(-5%, 2%) scale(1.04); }
        }
        @keyframes ray {
          0%, 100% { opacity: 0.6; transform: skewX(-5deg) translateX(0); }
          50% { opacity: 1; transform: skewX(-8deg) translateX(10px); }
        }
      `}</style>
    </div>
  );
}