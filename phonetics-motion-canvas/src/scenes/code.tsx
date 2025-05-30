import { Latex, makeScene2D, Node, Rect, Txt, TxtProps } from "@motion-canvas/2d";
import { all, createRef, delay, sequence, spring, waitFor } from "@motion-canvas/core";
import { TEXT_FILL, TEXT_FILL_FADED_SLIGHTLY, TEXT_FONT } from "./globals";
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
      x={0}
      y={450}
    />,
  );
  const matrix = new Matrix(matrixContainer);

  yield* all(
    wordsContainer.scale(0.8, 1.5),
    wordsContainer.y(-700, 1.5),
    matrixContainer().y(matrixContainer().y() - 200, 1.5),
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

  let duration = 1.0;
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

  yield* waitFor(2);
});
