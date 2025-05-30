import { Latex, makeScene2D, Node, Txt, TxtProps } from "@motion-canvas/2d";
import { TEXT_FILL, TEXT_FILL_FADED_SLIGHTLY, TEXT_FONT } from "./globals";

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
    <Node />
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
});
