import { Latex, makeScene2D, Node, Rect, Txt, TxtProps } from "@motion-canvas/2d";
import {
  all, chain, createRef, delay, sequence, spring,
  ThreadGenerator, waitFor,
} from "@motion-canvas/core";
import { HIGHLIGHT_COLOR_3, TEXT_FILL, TEXT_FILL_FADED_SLIGHTLY, TEXT_FONT } from "./globals";
import { Highlight } from "./Highlight";
import { Matrix } from "./Matrix";

const segmenter = new Intl.Segmenter("en", { granularity: "grapheme" });

// projection resolution: 1920x2160 (half screen in 4K)
export default makeScene2D(function* (view) {
  const word1 = Array.from(segmenter.segment("pɥisɑ̃s"), segment => segment.segment);
  const word2 = Array.from(segmenter.segment("nɥɑ̃s"), segment => segment.segment);

  // 🎈 Words as integer list & phonetic symbols as helpers

  const words = `\\begin{align}
  a &= \\bigl[ \\: 0, \\: 1, \\: 2, \\: 3, \\: 4, \\: 3 \\: \\bigr] \\\\
  b &= \\bigl[ \\: 5, \\: 1, \\: 4, \\: 3 \\: \\bigr]
  \\end{align}
  `;

  const wordsContainer = (
    <Node opacity={0} />
  ) as Node;
  view.add(wordsContainer);

  let textProps = {
    fill: TEXT_FILL,
    fontSize: 69,
  } as TxtProps;

  const wordsTxt = (
    <Latex
      tex={words}
      {...textProps}
    />
  ) as Latex;
  wordsContainer.add(wordsTxt);

  textProps.fontFamily = TEXT_FONT;
  const word1Symbols = word1.map((symbol, i) => (
    <Txt
      {...textProps}
      fill={TEXT_FILL_FADED_SLIGHTLY}
      y={-150}
      x={-160 + i * 95}
    >
      {symbol}
    </Txt>) as Txt,
  );
  const word2Symbols = word2.map((symbol, i) => (
    <Txt
      {...textProps}
      fill={TEXT_FILL_FADED_SLIGHTLY}
      y={150}
      x={-160 + i * 95}
    >
      {symbol}
    </Txt>) as Txt,
  );
  wordsContainer.add([...word1Symbols, ...word2Symbols]);

  const InboundSpring = {
    mass: 0.04,
    stiffness: 3.0,
    damping: 0.3,
    initialVelocity: 10.0,
  };
  const springAnim = spring(InboundSpring, -800, 0, (value) => {
    wordsContainer.x(value);
  });
  yield* all(
    springAnim,
    wordsContainer.opacity(1, 0.4),
  );

  yield* waitFor(0.2);

  // 🎈 Initialize matrix

  const matrixContainer = createRef<Rect>();
  view.add(
    <Rect
      ref={matrixContainer}
      x={-60}
      y={450}
    />,
  );
  const matrix = new Matrix(matrixContainer);

  yield* all(
    wordsContainer.scale(0.8, 1.5),
    wordsContainer.y(-700, 1.5),
    matrixContainer().y(matrixContainer().y() - 250, 1.5),
    delay(0.18,
      sequence(0.025, ...matrix.rects.map(rect => rect.opacity(1, 1))),
    ),
    delay(0.8,
      all(
        ...matrix.word1Texts.map(text => text.opacity(1, 1.5)),
        ...matrix.word2Texts.map(text => text.opacity(1, 1.5)),
      ),
    ),
  );

  // Transform IPA symbols to general variables a_0, a_1, ...
  delete textProps.fontFamily;
  const createVars = (texts: Txt[], prefix: string) => {
    const vars: Latex[] = [];
    for (let i = 0; i < texts.length; i++) {
      const latex = (
        <Latex
          {...textProps}
          tex={`${prefix}_{${i}}`}
          opacity={0}
        />
      ) as Latex;
      vars.push(latex);
      matrix.container().add(latex);
      latex.absolutePosition(texts[i].absolutePosition());
    }
    return vars;
  };
  const aVars = createVars(matrix.word1Texts, "a");
  const bVars = createVars(matrix.word2Texts, "b");
  for (const latex of [...aVars, ...bVars]) {
    latex.scale(0.7);
    latex.x(latex.x() - 70);
    latex.fill(TEXT_FILL_FADED_SLIGHTLY);
  }

  yield* waitFor(0.5);

  let duration = 1.1;
  yield* all(
    ...[...aVars, ...bVars].map((latex) => {
      return all(
        latex.opacity(1, duration),
        latex.x(latex.x() + 70, duration),
        latex.scale(1, duration),
        latex.fill(TEXT_FILL, duration),
      );
    }),
    ...[...matrix.word1Texts, ...matrix.word2Texts].map((text) => {
      return all(
        text.opacity(0, duration),
        text.fill(TEXT_FILL_FADED_SLIGHTLY, duration),
        text.scale(0.7, duration),
      );
    }),
  );

  // 🎈 Padding & first row/column (gap penalties)

  yield* matrix.highlight(0, 0, 1.0);
  yield* waitFor(0.5);
  const firstRect = matrix.getRectAt(0, 0);

  duration = 1.0;
  const fontSize = 53;
  yield* all(
    firstRect.fill(null, duration),
    matrix.writeTextAt(0, 0, "0", duration, false, fontSize),
    sequence(0.15,
      matrix.writeTextAt(0, 1, "-2", duration, false, fontSize),
      matrix.writeTextAt(0, 2, "-4", duration, false, fontSize),
      matrix.writeTextAt(0, 3, "-6", duration, false, fontSize),
      matrix.writeTextAt(0, 4, "-8", duration, false, fontSize),
    ),
    sequence(0.15,
      matrix.writeTextAt(1, 0, "-2", duration, false, fontSize),
      matrix.writeTextAt(2, 0, "-4", duration, false, fontSize),
      matrix.writeTextAt(3, 0, "-6", duration, false, fontSize),
      matrix.writeTextAt(4, 0, "-8", duration, false, fontSize),
      matrix.writeTextAt(5, 0, "-10", duration, false, fontSize),
      matrix.writeTextAt(6, 0, "-12", duration, false, fontSize),
    ),
  );

  yield* waitFor(0.5);

  // 🎈 Highlight all remaining matrix fields (nested for-loop)

  const highlightAnims: ThreadGenerator[] = [];

  const unfill = (i: number, j: number, duration: number) => {
    return matrix.getRectAt(i, j).fill(null, duration);
  };

  duration = 0.6;
  for (let i = 1; i <= word1.length; i++) {
    for (let j = 1; j <= word2.length; j++) {
      highlightAnims.push(chain(
        matrix.highlight(i, j, duration),
        unfill(i, j, duration),
      ));
    }
  }
  yield* sequence(0.07, ...highlightAnims);

  yield* waitFor(0.5);

  // 🎈 Consider directions for one field (local maximization)

  const theField = matrix.getRectAt(1, 1);
  yield* theField.stroke(HIGHLIGHT_COLOR_3, 0.8);

  duration = 0.9;
  const [arrowFromLeftAnim, arrowFromLeft] = matrix.stepAndArrowStay(
    1, 0, 1, 1, duration, -18, true, HIGHLIGHT_COLOR_3, true);
  yield* arrowFromLeftAnim;

  yield* waitFor(0.5);

  const [arrowFromTopAnim, arrowFromTop] = matrix.stepAndArrowStay(
    0, 1, 1, 1, duration, -18, true, HIGHLIGHT_COLOR_3, true);
  yield* arrowFromTopAnim;

  yield* waitFor(0.5);

  const [arrowFromDiagonalAnim, arrowFromDiagonal] = matrix.stepAndArrowStay(
    0, 0, 1, 1, duration, -18, true, HIGHLIGHT_COLOR_3, true);
  yield* arrowFromDiagonalAnim;

  yield* waitFor(0.5);

  // 🎈 Match/Mismatch score
  const highlightA0 = (
    <Highlight
      stroke={HIGHLIGHT_COLOR_3}
      width={110}
      height={110}
      x={-450 / 2}
      y={-30 / 2}
    />
  ) as Highlight;
  const highlightB0 = (
    <Highlight
      stroke={HIGHLIGHT_COLOR_3}
      width={110}
      height={110}
      x={-140 / 2}
      y={-330 / 2}
    />
  ) as Highlight;
  view.add([highlightA0, highlightB0]);
  duration = 1.2;
  aVars[0].shadowColor(HIGHLIGHT_COLOR_3);
  bVars[0].shadowColor(HIGHLIGHT_COLOR_3);
  yield* all(
    highlightA0.highlight(duration),
    highlightB0.highlight(duration),
    aVars[0].shadowBlur(30, duration * 2),
    bVars[0].shadowBlur(30, duration * 2),
  );

  // 🎈 One last time every field again
  const highlightFinalAnims: ThreadGenerator[] = [];
  duration = 0.6;
  const fieldScores = [
    -1, -3, -5, -7,
    -3, 0, -2, -4,
    -5, -2, -1, -3,
    -7, -4, -3, 0,
    -9, -6, -3, -2,
    -11, -8, -5, -2,
  ];
  let counter = 0;
  for (let i = 1; i <= word1.length; i++) {
    for (let j = 1; j <= word2.length; j++) {
      const rect = matrix.getRectAt(i, j);
      highlightFinalAnims.push(chain(
        all(
          rect.fill(HIGHLIGHT_COLOR_3, duration),
          rect.stroke(HIGHLIGHT_COLOR_3, duration),
          matrix.writeTextAt(i, j, fieldScores[counter].toString(),
            duration, false, fontSize, false),
        ),
        unfill(i, j, duration),
      ));
      counter++;
    }
  }
  duration = 1.5;
  yield* all(
    arrowFromTop.opacity(0, duration),
    arrowFromLeft.opacity(0, duration),
    arrowFromDiagonal.opacity(0, duration),
    sequence(0.09, ...highlightFinalAnims),
  );

  yield* waitFor(2);
});
