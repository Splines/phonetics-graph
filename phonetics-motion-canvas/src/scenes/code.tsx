import { Latex, makeScene2D, Rect } from "@motion-canvas/2d";
import { TEXT_FILL } from "./globals";

// projection resolution: 1920x2160 (half screen in 4K)
export default makeScene2D(function* (view) {
  const words = `\\begin{align}
  a &= \\bigl[0, 1, 2, 3, 4, 3\\bigr] \\\\
  b &= \\bigl[5, 1, 4, 3\\bigr]
  \\end{align}
  `;

  const wordsContainer = (
    <Rect
      layout
      direction="column"
      alignItems="start"
    />
  ) as Rect;
  view.add(wordsContainer);

  let textProps = {
    fill: TEXT_FILL,
    fontSize: 80,
  };

  const wordsTxt = (
    <Latex
      tex={words}
      {...textProps}
    />
  ) as Latex;
  wordsContainer.add(wordsTxt);
});
