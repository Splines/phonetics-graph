import { Latex, makeScene2D } from "@motion-canvas/2d";
import { waitFor } from "@motion-canvas/core";

export default makeScene2D(function* (view) {
  const latex = (
    <Latex
      tex="{{a^2}} + {{b^2}}"
      fontSize={100}
      fill="#FFFFFF"
    />
  ) as Latex;
  view.add(latex);

  const latex2 = latex.snapshotClone();
  view.add(latex2);
  latex2.position.y(500);

  yield* latex2.tex("{{a^2}} + {{b^2}} = c^2", 1.0); // 😕
  yield* waitFor(0.4);
  yield* latex.tex("{{a^2}} + {{b^2}} = c^2", 1.0); // ✨
  yield* waitFor(1);
});
