import { is, Line, makeScene2D, Node, Rect, Txt } from "@motion-canvas/2d";
import {
  all, chain, createRef, delay,
  sequence, ThreadGenerator,
  waitFor,
} from "@motion-canvas/core";
import { AlignState, generateAllPossibleAlignmentStrings } from "./AlignState";
import { Highlight } from "./Highlight";
import { LetterTxt } from "./LetterTxt";
import { Matrix } from "./Matrix";
import { HIGHLIGHT_COLOR, HIGHLIGHT_COLOR_2, TEXT_FILL, TEXT_FONT } from "./globals";

export default makeScene2D(function* (view) {
  const container = createRef<Node>();
  const alignState = new AlignState(container, "pɥisɑ̃s", "nɥɑ̃s", "....--");
  const texts = alignState.generateElements();
  const alignmentDelta = 100;
  for (const text of texts) {
    text.position.y(text.position.y() + alignmentDelta);
    text.opacity(0);
  }

  view.add(
    <Rect ref={container}>
      { texts }
    </Rect>,
  );

  yield* waitFor(0.1);

  yield* all(
    ...texts.map(txt => txt.position.y(txt.position.y() - alignmentDelta, 0.8)),
    ...texts.map(txt => txt.opacity(1, 0.8)),
  );

  yield* waitFor(1);

  yield* sequence(0.2, ...[0, 1, 2, 3, 4, 5, 6, 7].map((i) => {
    return texts[i].fill(HIGHLIGHT_COLOR, 0.6);
  }));

  yield* waitFor(0.5);

  yield* sequence(0.2, ...[8, 9, 10, 11].map((i) => {
    return texts[i].fill(HIGHLIGHT_COLOR_2, 0.6);
  }));

  yield* waitFor(0.5);

  // 🎈 All possible alignments
  let alignmentStrings = generateAllPossibleAlignmentStrings(6, 4);
  const riseAround = alignmentStrings.length * 0.999;
  const c = 0.13;

  // for (let i = 0; i < alignmentStrings.length; i++) {
  //   const fall = Math.exp(-0.04 * i);
  //   const exp = Math.exp(c * (i - riseAround));
  //   const rise = exp / (1 + exp);
  //   const stretch = Math.max(fall + rise, 0.006);

  //   if (i === 0) {
  //     yield* all(
  //       alignState.animateToState(alignmentStrings[i], 0.5 * stretch),
  //       all(...texts.map(txt => txt.fill(TEXT_FILL, 0.5))),
  //     );
  //   } else {
  //     yield* alignState.animateToState(alignmentStrings[i], 0.5 * stretch);
  //   }
  //   yield* waitFor(0.3 * stretch);
  // }

  yield* waitFor(0.2);
  yield* alignState.animateToState("....--", 1);

  // 🎈 Discuss one alignment
  yield* waitFor(0.5);
  yield* alignState.animateToState("..--..", 1.4);

  yield* waitFor(1);

  yield* sequence(0.2, ...[2, 3, 8, 5, 10, 7].map((i) => {
    return texts[i].fill(HIGHLIGHT_COLOR, 0.6);
  }));

  yield* waitFor(1);

  yield* sequence(0.2, ...[0, 1].map((i) => {
    return texts[i].fill(HIGHLIGHT_COLOR_2, 0.6);
  }));

  yield* waitFor(1);

  const textsAll = view.findAll(is(Txt));
  yield* sequence(0.2, ...[4, 10, 6, 11].map((i) => {
    return textsAll[i].fill(HIGHLIGHT_COLOR_2, 0.6);
  }));

  yield* waitFor(1);

  const rect = createRef<Rect>();
  view.add(
    <Rect
      ref={rect}
      rotation={180}
      lineWidth={5}
      stroke={TEXT_FILL}
      width={780}
      height={350}
      radius={6}
      closed
      opacity={0.7}
      end={0}
    />,
  );
  yield* all(
    rect().end(1, 0.8),
    all(...textsAll.map(txt => txt.fill(TEXT_FILL, 0.8))),
  );
  yield* waitFor(1.6);
  yield* rect().opacity(0, 1.2);
  yield* waitFor(0.2);

  // 🎈 Random alignments
  // const tenRandomNumbers = Array.from({ length: 10 },
  //   () => Math.floor(Math.random() * alignmentStrings.length));
  // for (const i of tenRandomNumbers) {
  //   console.log(alignmentStrings[i]);
  // }
  const randomAlignmentStrings = ["-.--:.-:", ".-:-.-:-", ":---:--.:",
    ":-.--::--", "....--"];
  const textTransforms = randomAlignmentStrings.map((str) => {
    return chain(
      alignState.animateToState(str, 0.9),
      waitFor(0.3),
    );
  });

  const toMatrixLine = createRef<Line>();
  view.add(
    <Line
      ref={toMatrixLine}
      points={[
        [-150, 0],
        [150, 0],
      ]}
      lineWidth={10}
      stroke={TEXT_FILL}
      arrowSize={25}
      endArrow
      lineCap="round"
      lineDash={[20, 20]}
      opacity={0}
      end={0}
    />,
  );

  const pathText = (
    <LetterTxt
      fill={TEXT_FILL}
      fontFamily={TEXT_FONT}
      fontSize={100}
      letterSpacing={4}
      x={360}
    >
      Path in a grid
    </LetterTxt>
  ) as LetterTxt;
  view.add(pathText);

  yield* all(
    chain(waitFor(0.8), ...textTransforms),
    container().position.x(-800, 5),
    chain(waitFor(2.3), all(toMatrixLine().opacity(1, 0.5), toMatrixLine().end(1, 3))),
    chain(waitFor(4), pathText.flyIn(1, 0.035)),
  );

  yield* waitFor(0.5);

  // 🎈 Construct the matrix
  const matrixContainerRef = createRef<Node>();
  view.add(
    <Rect
      ref={matrixContainerRef}
      x={520}
      y={-65}
    />,
  );
  const matrix = new Matrix(matrixContainerRef);

  yield* all(
    toMatrixLine().opacity(0, 0.8),
    pathText.opacity(0, 0.8),
    sequence(0.03, ...matrix.rects.map(rect => rect.opacity(1, 1))),
  );

  // Text to new positions
  const animateTextClones = (indices: number[], targetTexts: Txt[]) => {
    const anims: ThreadGenerator[] = [];

    let iTarget = 0;
    for (const i of indices) {
      const original = texts[i];
      const target = targetTexts[iTarget++];

      const clone = original.clone();
      clone.opacity(0);
      view.add(clone);
      clone.absolutePosition(original.absolutePosition());

      anims.push(
        chain(
          all(
            clone.absolutePosition(target.absolutePosition(), 0.7),
            clone.opacity(1, 0.1),
            original.opacity(0, 0.1),
            clone.fontSize(target.fontSize(), 0.7),
          ),
          all(
            // replace clone by target
            clone.opacity(0, 0),
            target.opacity(1, 0),
          ),
        ),
      );
    }

    return anims;
  };

  const gaps = view.findAll(is(Txt)).filter(txt => txt.text() === "–");

  yield* sequence(0.13, ...animateTextClones([0, 2, 4, 6, 8, 10], matrix.word1Texts));
  yield* waitFor(0.3);
  yield* all(
    sequence(0.13, ...animateTextClones([1, 3, 5, 7], matrix.word2Texts)),
    ...gaps.map(txt => txt.opacity(0, 1.2)),
  );

  yield* waitFor(1);

  container().x(container().x() + 300);

  // 🎈 Diagonal steps
  const animateMatrixStep = function* (
    startRow: number,
    startCol: number,
    endRow: number,
    endCol: number,
    highlightIndices: number[],
    duration: number,
    customTexts?: Txt[],
  ) {
    yield* matrix.step(startRow, startCol, endRow, endCol, duration);
    yield* matrix.highlightCoordinates(endRow, endCol, duration);

    let txt = customTexts ? customTexts : texts;
    yield* all(
      ...highlightIndices.map(i => txt[i].opacity(1, duration)),
      ...highlightIndices.map(i => txt[i].fill(HIGHLIGHT_COLOR, duration)),
    );
    yield* waitFor(0.6);
    yield* all(
      ...highlightIndices.map(i => txt[i].fill(TEXT_FILL, duration)),
    );
  };

  yield* matrix.highlight(0, 0, 0.8);
  yield* waitFor(1.5);
  yield* animateMatrixStep(0, 0, 1, 1, [0, 1], 0.8);
  yield* animateMatrixStep(1, 1, 2, 2, [2, 3], 0.8);
  yield* animateMatrixStep(2, 2, 3, 3, [4, 5], 0.8);
  yield* animateMatrixStep(3, 3, 4, 4, [6, 7], 0.8);
  yield* all(
    matrix.word1Texts[3].fill(TEXT_FILL, 0.8),
    matrix.word2Texts[3].fill(TEXT_FILL, 0.8),
  );

  // 🎈 Missing letters in puissance
  const highlight = (
    <Highlight
      width={420}
      height={130}
      x={-300}
      y={-28}
    />
  ) as Highlight;
  view.add(highlight);
  yield* highlight.highlight(1);

  yield* waitFor(1.0);

  yield* sequence(0.5, ...[8, 10].map((i) => {
    return all(
      texts[i].opacity(1, 0.6),
      texts[i].fill(HIGHLIGHT_COLOR_2, 0.6),
    );
  }));
  yield* waitFor(0.8);
  yield* sequence(0.5, ...gaps.map((txt) => {
    return all(
      txt.opacity(1, 0.6),
      txt.fill(HIGHLIGHT_COLOR_2, 0.6),
    );
  }));
  yield* waitFor(0.2);

  yield* all(
    texts[8].fill(TEXT_FILL, 0.8),
    texts[10].fill(TEXT_FILL, 0.8),
    ...gaps.map(txt => txt.fill(TEXT_FILL, 0.8)),
  );

  yield* animateMatrixStep(4, 4, 5, 4, [], 0.8);
  yield* all(
    texts[8].fill(HIGHLIGHT_COLOR, 0.8),
    gaps[0].fill(HIGHLIGHT_COLOR, 0.8),
  );
  yield* waitFor(0.5);
  yield* all(
    texts[8].fill(TEXT_FILL, 0.8),
    gaps[0].fill(TEXT_FILL, 0.8),
  );
  yield* animateMatrixStep(5, 4, 6, 4, [], 0.8);
  yield* all(
    texts[10].fill(HIGHLIGHT_COLOR, 0.8),
    gaps[1].fill(HIGHLIGHT_COLOR, 0.8),
  );
  yield* waitFor(0.5);
  yield* all(
    texts[10].fill(TEXT_FILL, 1.2),
    gaps[1].fill(TEXT_FILL, 1.2),
    matrix.word1Texts[5].fill(TEXT_FILL, 1.2),
    matrix.word2Texts[3].fill(TEXT_FILL, 1.2),
  );
  yield* waitFor(1);

  // 🎈 Go back some steps
  let duration = 0.5;
  yield* all(
    matrix.getRectAt(6, 4).stroke("white", duration * 1.2),
    matrix.step(6, 4, 5, 4, duration),
    texts[10].opacity(0, duration * 1.5),
    gaps[1].opacity(0, duration * 1.5),
  );
  yield* all(
    matrix.getRectAt(5, 4).stroke("white", duration * 1.2),
    matrix.step(5, 4, 4, 4, duration),
    texts[8].opacity(0, duration * 1.5),
    gaps[0].opacity(0, duration * 1.5),
  );
  yield* all(
    matrix.getRectAt(4, 4).stroke("white", duration * 1.2),
    matrix.step(4, 4, 3, 3, duration),
    texts[6].opacity(0, duration * 1.5),
    texts[7].opacity(0, duration * 1.5),
  );

  yield* waitFor(1);

  const container2 = createRef<Node>();
  const alignment2 = new AlignState(container2, "pɥisɑ̃s", "nɥɑ̃s", "...:---");
  const texts2 = alignment2.generateElements();
  for (const text of texts2) {
    text.position.y(text.position.y() + alignmentDelta);
    text.opacity(0);
  }
  view.add(
    <Rect ref={container2} x={-450} y={-100}>
      { texts2 }
    </Rect>,
  );

  // 🎈 Take horizontal step instead & finish alignment
  duration = 0.4;
  yield* animateMatrixStep(3, 3, 3, 4, [6, 7], 1.0, texts2), // wait
  yield* animateMatrixStep(3, 4, 4, 4, [8, 9], duration, texts2),
  yield* animateMatrixStep(4, 4, 5, 4, [10, 11], duration, texts2),
  yield* animateMatrixStep(5, 4, 6, 4, [12, 13], duration, texts2),
  yield* all(
    matrix.word1Texts[5].fill(TEXT_FILL, 1.2),
    matrix.word2Texts[3].fill(TEXT_FILL, 1.2),
  );

  yield* waitFor(1);

  // 🎈 Go back some steps
  duration = 0.5;
  yield* all(
    matrix.getRectAt(6, 4).stroke("white", duration * 1.2),
    matrix.step(6, 4, 5, 4, duration),
    texts2[12].opacity(0, duration * 1.5),
    texts2[13].opacity(0, duration * 1.5),
  );
  yield* all(
    matrix.getRectAt(5, 4).stroke("white", duration * 1.2),
    matrix.step(5, 4, 4, 4, duration),
    texts2[10].opacity(0, duration * 1.5),
    texts2[11].opacity(0, duration * 1.5),
  );
  yield* all(
    matrix.getRectAt(4, 4).stroke("white", duration * 1.2),
    matrix.step(4, 4, 3, 4, duration),
    texts2[8].opacity(0, duration * 1.5),
    texts2[9].opacity(0, duration * 1.5),
  );
  yield* all(
    matrix.getRectAt(3, 4).stroke("white", duration * 1.2),
    matrix.step(3, 4, 3, 3, duration),
    texts2[6].opacity(0, duration * 1.5),
    texts2[7].opacity(0, duration * 1.5),
  );

  yield* waitFor(1);

  // 🎈 Take not-allowed step (bottom-left) & finish alignment
  const container3 = createRef<Node>();
  const alignment3 = new AlignState(container3, "pɥisɑ̃s", "nɥɑ̃ɥɑ̃s", "......");
  const texts3 = alignment3.generateElements();
  for (const text of texts3) {
    text.position.y(text.position.y() + alignmentDelta);
    text.opacity(0);
  }
  view.add(
    <Rect ref={container3} x={-500} y={-100}>
      { texts3 }
    </Rect>,
  );

  duration = 0.8;
  yield* animateMatrixStep(3, 3, 4, 2, [6, 7], duration, texts3);
  yield* waitFor(1);
  yield* animateMatrixStep(4, 2, 5, 3, [8, 9], duration, texts3);
  yield* animateMatrixStep(5, 3, 6, 4, [10, 11], duration, texts3);
  yield* all(
    matrix.word1Texts[5].fill(TEXT_FILL, 1.2),
    matrix.word2Texts[3].fill(TEXT_FILL, 1.2),
  );

  yield* waitFor(1);

  const highlightWeird = (
    <Highlight
      width={600}
      height={126}
      x={-252}
      y={30}
    />
  ) as Highlight;
  view.add(highlightWeird);
  yield* highlightWeird.highlight(1.2);

  yield* waitFor(1);

  // 🎈 Highlight first bad move
  duration = 1.0;
  const startRect = matrix.getRectAt(3, 3);
  yield* all(
    startRect.stroke(matrix.BASE_COLOR, duration),
    startRect.lineWidth(10, duration),
  );
  yield* matrix.step(3, 3, 4, 2, duration, true);

  yield* waitFor(1);

  // 🎈 Reset all matrix fields except (3,3) and (4,2) and move matrix to center
  duration = 2.0;

  let resetMatrix: ThreadGenerator[] = [];
  for (let row = 0; row < matrix.numRows; row++) {
    for (let col = 0; col < matrix.numCols; col++) {
      if ((row === 3 && col === 3) || (row === 4 && col === 2)) {
        continue;
      }
      const rect = matrix.getRectAt(row, col);
      resetMatrix.push(
        all(
          rect.fill(null, duration),
          rect.stroke(matrix.BASE_COLOR, duration),
        ),
      );
    }
  }
  yield* all(
    matrix.container().x(0, 2.5),
    ...texts.map(txt => txt.opacity(0, duration)),
    ...gaps.map(txt => txt.opacity(0, duration)),
    ...texts2.map(txt => txt.opacity(0, duration)),
    ...texts3.map(txt => txt.opacity(0, duration)),
    ...resetMatrix,
  );

  yield* waitFor(1);

  // 🎈 Highlight all bad & good moves
  const badMoveEnds = [[3, 2], [2, 2], [2, 3], [2, 4]];
  yield* chain(
    ...badMoveEnds.map(([row, col]) => {
      return matrix.step(3, 3, row, col, 0.8, true);
    }),
  );

  yield* waitFor(1);

  const goodMoveEnds = [[3, 4], [4, 4], [4, 3]];
  yield* chain(
    ...goodMoveEnds.map(([row, col]) => {
      return matrix.step(3, 3, row, col, 0.8);
    }),
  );

  yield* waitFor(1);

  // 🎈 Fade out matrix & show exercise alignment
  const containerExercise = createRef<Node>();
  const alignmentExercise = new AlignState(containerExercise, "pɥisɑ̃s", "nɥɑ̃s", "..--..");
  const textsAlignment = alignmentExercise.generateElements();
  for (const text of textsAlignment) {
    text.position.y(text.position.y() + 170);
    text.opacity(0);
  }
  view.add(
    <Rect ref={containerExercise} x={0} y={0}>
      { textsAlignment.map((txt) => {
        txt.rotation(20);
        return txt;
      })}
    </Rect>,
  );

  const highlightExercise = (
    <Highlight
      width={650}
      height={250}
      x={0}
      y={0}
    />
  ) as Highlight;
  view.add(highlightExercise);

  yield* all(
    ...matrix.word1Texts.map(txt => txt.opacity(0, 1.5)),
    ...matrix.word2Texts.map(txt => txt.opacity(0, 1.5)),
    sequence(0.02,
      ...matrix.rects.map((rect) => {
        return all(
          rect.height(0, 1),
          rect.opacity(0, 1),
        );
      }),
    ),
    delay(1, all(
      ...textsAlignment.map(txt => txt.position.y(txt.position.y() - 170, 1.2)),
      ...textsAlignment.map(txt => txt.opacity(1, 1.2)),
      ...textsAlignment.map(txt => txt.rotation(0, 1.2)),
    )),
    delay(1.6, highlightExercise.highlight(1.2)),
  );

  yield* waitFor(2);
});
