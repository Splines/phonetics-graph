import { Latex, makeScene2D } from "@motion-canvas/2d";
import { all, Spring, spring, waitFor } from "@motion-canvas/core";
import { TEXT_FILL } from "./globals";

export default makeScene2D(function* (view) {
  const needlemanWunschTxt = (
    <Latex
      tex="\mathcal{O}\Bigl(\text{len}(a) \cdot \text{len}(b)\Bigr)"
      fontSize={200}
      fill={TEXT_FILL}
      x={-1000}
      y={0}
      opacity={0}
    />
  ) as Latex;
  view.add(needlemanWunschTxt);

  const TextSpring: Spring = {
    mass: 0.07,
    stiffness: 5.2,
    damping: 0.4,
  };
  yield* all(
    needlemanWunschTxt.opacity(1, 0.5),
    spring(TextSpring, -1000, 0, 1, (value) => {
      needlemanWunschTxt.x(value);
    }),
  );

  yield* waitFor(1);
});
