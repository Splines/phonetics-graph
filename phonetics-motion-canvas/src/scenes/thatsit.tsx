import { makeScene2D, Txt } from "@motion-canvas/2d";
import { all, Spring, spring, waitFor } from "@motion-canvas/core";
import { TEXT_FILL, TEXT_FONT } from "./globals";

export default makeScene2D(function* (view) {
  const needlemanWunschTxt = (
    <Txt
      fontFamily={TEXT_FONT}
      fontSize={200}
      fill={TEXT_FILL}
      x={0}
      y={1000}
      opacity={0}
    >
      Needleman–Wunsch
    </Txt>
  ) as Txt;
  view.add(needlemanWunschTxt);

  const TextSpring: Spring = {
    mass: 0.1,
    stiffness: 5.0,
    damping: 0.4,
  };
  yield* all(
    needlemanWunschTxt.opacity(1, 0.5),
    spring(TextSpring, 1000, 0, 1, (value) => {
      needlemanWunschTxt.y(value);
    }),
  );

  yield* waitFor(1);
});
