import { is, Latex, Line, makeScene2D, Node, Rect, Txt } from "@motion-canvas/2d";
import {
  all,
  chain,
  createRef, delay, sequence, Spring,
  spring, waitFor,
} from "@motion-canvas/core";
import { AlignState } from "./AlignState";
import {
  HIGHLIGHT_COLOR,
  HIGHLIGHT_COLOR_2,
  TEXT_FILL, TEXT_FILL_DARK, TEXT_FONT,
} from "./globals";
import { Highlight } from "./Highlight";
import { LetterTxt } from "./LetterTxt";
import { Matrix } from "./Matrix";
import { ScoreRuler } from "./ScoreRuler";

export default makeScene2D(function* (view) {
  let duration = 0.8;

  // 🎈 Alignment -> Path in matrix
  const alignmentContainer = createRef<Node>();
  const alignment = new AlignState(alignmentContainer, "pɥisɑ̃s", "nɥɑ̃s", "..--..");
  const texts = alignment.generateElements();

  view.add(
    <Rect ref={alignmentContainer}>
      { texts }
    </Rect>,
  );

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

  const matrixContainer = createRef<Node>();
  view.add(
    <Rect
      ref={matrixContainer}
      x={520}
      y={-65}
    />,
  );
  const matrix = new Matrix(matrixContainer);

  let alignmentString = ":-.:-:---";
  yield* alignment.animateToState(alignmentString, 1.2);

  const TextSpring: Spring = {
    mass: 0.13,
    stiffness: 4.3,
    damping: 0.5,
  };

  const extendSpring = 80;
  yield* alignmentContainer().x(extendSpring, 0.7);
  yield* all(
    all(
      spring(TextSpring, extendSpring, -750, 1, (value) => {
        alignmentContainer().x(value);
      }),
      matrixContainer().x(750, 1.5),
    ),
    delay(0.2, all(
      all(
        toMatrixLine().opacity(1, 1.0),
        toMatrixLine().end(1, 1.5),
      ),
      sequence(0.03, ...matrix.rects.map(rect => rect.opacity(1, 1))),
      delay(0.8,
        all(
          ...matrix.word1Texts.map(text => text.opacity(1, 1.5)),
          ...matrix.word2Texts.map(text => text.opacity(1, 1.5)),
        ),
      ),
    )),
    delay(1.4, sequence(0.1,
      ...matrix.highlightAlignmentPath(alignmentString, 0.4),
    )),
  );

  yield* waitFor(1.5);

  // 🎈 Path in matrix -> score
  const matrixToScoreLine = createRef<Line>();
  view.add(
    <Line
      ref={matrixToScoreLine}
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
      x={850}
    />,
  );

  const scoreRuler = (
    <ScoreRuler
      points={[
        [0, 400],
        [0, -400],
      ]}
      x={570}
      maxValue={30}
    />
  ) as ScoreRuler;
  view.add(scoreRuler);
  scoreRuler.ruler.end(0);
  scoreRuler.value(-30);

  yield* all(
    sequence(0.06,
      alignmentContainer().x(-1200, 1.5),
      toMatrixLine().x(-500, 1.5),
      matrixContainer().x(150, 1.5),
    ),
    delay(0.2,
      all(
        matrixToScoreLine().opacity(1, 1.0),
        matrixToScoreLine().end(1, 1.5),
      ),
    ),
    delay(0.5, all(
      scoreRuler.opacity(1, 1),
      scoreRuler.ruler.end(1, 1),
      scoreRuler.value(30, 1.4),
    )),
  );

  yield* scoreRuler.value(-15, 1.2);

  yield* waitFor(1.0);

  const resetMatrixRects = (duration: number) => matrix.rects.map((rect) => {
    return all(
      rect.lineWidth(6, duration),
      rect.stroke(TEXT_FILL, duration),
    );
  });

  alignmentString = "..--..";
  yield* all(
    all(
      alignment.animateToState(alignmentString, 1.6),
      alignmentContainer().x(-1050, 1.6),
      ...resetMatrixRects(1.2),
    ),
    delay(1.5, sequence(0.1,
      ...matrix.highlightAlignmentPath(alignmentString, 0.4),
    )),
    delay(1.9, all(
      scoreRuler.value(-2, 1.5),
    )),
  );

  yield* waitFor(2);

  // 🎈 Bad gaps
  alignmentString = ":-.:-:---";

  yield* all(
    alignmentContainer().x(0, 2.0),
    alignment.animateToState(alignmentString, 1.8),
    all(
      matrixContainer().x(600, 1.8),
      matrixContainer().opacity(0, 1.7),
      matrixToScoreLine().x(1100, 1.7),
      matrixToScoreLine().opacity(0, 1.7),
      scoreRuler.x(700, 1.7),
      scoreRuler.opacity(0, 1.7),
      ...resetMatrixRects(2.0),
      toMatrixLine().x(-100, 1.7),
      toMatrixLine().opacity(0, 1.7),
    ),
  );

  yield* waitFor(1);

  const gaps = view.findAll(is(Txt)).filter(txt => txt.text() === "–");
  gaps.sort((a, b) => a.x() - b.x());

  // Construct highlight boxes around each gap
  const highlightBoxes = gaps.map((txt) => {
    const highlight = (
      <Highlight
        width={70}
        height={50}
      />
    ) as Highlight;
    const newCoord = txt.localToParent().transformPoint();
    highlight.absolutePosition(newCoord);
    highlight.y(highlight.y() + 10);

    return highlight;
  });
  alignmentContainer().add(highlightBoxes);

  yield* sequence(0.3,
    ...gaps.map((text, i) => {
      return all(
        text.fill(HIGHLIGHT_COLOR, 0.8),
        highlightBoxes[i].highlight(0.8),
      );
    }));

  yield* waitFor(1);

  const gapPenaltyTxt = (
    <LetterTxt
      fill={TEXT_FILL}
      fontFamily={TEXT_FONT}
      fontSize={108}
      letterSpacing={4}
      x={200}
    >
      Gap Penalty
    </LetterTxt>
  ) as LetterTxt;
  const constP = (
    <Latex
      tex="{{p}} \in \mathbb{R}"
      fill={TEXT_FILL}
      fontSize={100}
      x={970}
      opacity={0}
    />
  ) as Latex;
  view.add([gapPenaltyTxt, constP]);

  yield* all(
    alignmentContainer().x(alignmentContainer().x() - 600, 1.5),
    delay(0.3, all(
      gapPenaltyTxt.flyIn(1.0, 0.02),
      delay(0.5, constP.opacity(1, 1.0)),
    )),
  );

  yield* waitFor(1);

  // 🎈 Gap penalty in matrix
  const newRectWidth = 150;
  matrixContainer().x(150);
  yield* all(
    matrixContainer().opacity(1, 1.3),
    matrixContainer().x(-400, 1.3),
    gapPenaltyTxt.x(gapPenaltyTxt.x() + 100, 1.3),
    constP.x(constP.x() + 100, 1.3),
    alignmentContainer().x(alignmentContainer().x() - 300, 1.3),
    alignmentContainer().opacity(0, 1.3),
    delay(0.5, all(
      matrix.layout().width(1100, 1.2),
      ...matrix.getAllRects().map((rect) => {
        return all(
          rect.width(newRectWidth, 1.2),
          rect.height(newRectWidth, 1.2));
      }),
    )),
  );

  yield* waitFor(1);

  yield* matrix.writeTextAt(0, 0, "0", 0.8);
  yield* waitFor(1);
  yield* matrix.step(0, 0, 0, 1, 0.8);
  yield* waitFor(2);

  const pOnly = (
    <Latex
      tex="p"
      fill={TEXT_FILL}
      fontSize={100}
      x={954}
      y={14}
      opacity={1}
    />
  ) as Latex;
  view.add(pOnly);

  yield* all(
    pOnly.fontSize(61, 1.2),
    pOnly.position([-535, -490], 1.2),
    delay(0.2, pOnly.fill(TEXT_FILL_DARK, 1.4)),
  );

  yield* all(
    matrix.writeTextAt(0, 1, "p", 0),
    pOnly.opacity(0, 0),
  );

  const highlightMinus2 = (
    <Highlight
      width={400}
      height={130}
      x={550}
      y={-5}
    />
  ) as Highlight;
  view.add(highlightMinus2);
  const matrixPLatex = matrix.getRectAt(0, 1).children()[0] as Latex;
  yield* all(
    constP.tex("{{p}} = -2", 1.3),
    constP.x(1100, 1.3),
    delay(0.4, all(
      matrixPLatex.tex("-2", 1.3),
      highlightMinus2.highlight(0.9),
    )),
  );

  yield* waitFor(2);

  // 🎈 Finish all first gap steps (horizontal, then vertical)
  yield* all(
    matrix.step(0, 1, 0, 2, 0.7),
    delay(0.8, matrix.writeTextAt(0, 2, "-4", 0.6)),
  );
  yield* all(
    matrix.step(0, 2, 0, 3, 0.7),
    delay(0.8, matrix.writeTextAt(0, 3, "-6", 0.6)),
  );
  yield* all(
    matrix.step(0, 3, 0, 4, 0.7),
    delay(0.8, matrix.writeTextAt(0, 4, "-8", 0.6)),
  );

  yield* waitFor(0.3);
  let latestRect = matrix.getRectAt(0, 4);
  yield* all(
    latestRect.lineWidth(7, 0.8),
    // latestRect.stroke(matrix.BASE_COLOR, 0.8),
    latestRect.fill(null, 0.8),
    (latestRect.children()[0] as Latex).fill(TEXT_FILL, 0.8),
  );

  yield* waitFor(1);

  // go down
  yield* all(
    matrix.step(0, 0, 1, 0, 0.7),
    delay(0.8, matrix.writeTextAt(1, 0, "-2", 0.6)),
  );
  yield* all(
    matrix.step(1, 0, 2, 0, 0.7),
    delay(0.8, matrix.writeTextAt(2, 0, "-4", 0.6)),
  );
  yield* all(
    matrix.step(2, 0, 3, 0, 0.7),
    delay(0.8, matrix.writeTextAt(3, 0, "-6", 0.6)),
  );
  yield* all(
    matrix.step(3, 0, 4, 0, 0.7),
    delay(0.8, matrix.writeTextAt(4, 0, "-8", 0.6)),
  );
  yield* all(
    matrix.step(4, 0, 5, 0, 0.7),
    delay(0.8, matrix.writeTextAt(5, 0, "-10", 0.6)),
  );
  yield* all(
    matrix.step(5, 0, 6, 0, 0.7),
    delay(0.8, matrix.writeTextAt(6, 0, "-12", 0.6)),
  );

  yield* waitFor(0.15);
  latestRect = matrix.getRectAt(6, 0);
  yield* all(
    latestRect.lineWidth(7, 1.3),
    latestRect.fill(null, 1.3),
    (latestRect.children()[0] as Latex).fill(TEXT_FILL, 1.3),
  );

  yield* waitFor(1);

  // 🎈 Diagonal steps with emojis

  const diagonalStepTxt = (
    <LetterTxt
      fill={TEXT_FILL}
      fontFamily={TEXT_FONT}
      fontSize={108}
      letterSpacing={4}
      x={300}
    >
      Diagonal Steps
    </LetterTxt>
  ) as LetterTxt;
  view.add(diagonalStepTxt);

  yield* all(
    gapPenaltyTxt.opacity(0, 1.0),
    constP.opacity(0, 1.0),
    diagonalStepTxt.flyIn(1.0, 0.02),
  );

  yield* all(
    matrix.step(0, 0, 1, 1, 0.7),
    delay(1.0, matrix.writeTextAt(1, 1, "\\textbf{?}", 0.6)),
  );

  yield* waitFor(1);
  yield* matrix.highlightCoordinates(1, 1, 1.0);
  yield* waitFor(1);
  yield* all(
    matrix.step(1, 1, 2, 2, 0.7),
    delay(0.7, matrix.writeTextAt(2, 2, "\\textbf{?}", 0.6)),
  );
  yield* matrix.highlightCoordinates(2, 2, 1.0);

  yield* waitFor(1);

  const badAlignRect = matrix.getRectAt(1, 1) as Rect;
  const goodAlignRect = matrix.getRectAt(2, 2) as Rect;
  const goodAlignEmoji = (
    <Txt
      fill={TEXT_FILL}
      fontSize={70}
      opacity={0}
    >
      🤩
    </Txt>
  ) as Txt;
  const badAlignEmoji = (
    <Txt
      fill={TEXT_FILL}
      fontSize={70}
      opacity={0}
    >
      😕
    </Txt>
  ) as Txt;
  view.add([badAlignEmoji, goodAlignEmoji]);
  goodAlignEmoji.absolutePosition(goodAlignRect.absolutePosition());
  badAlignEmoji.absolutePosition(badAlignRect.absolutePosition());

  yield* all(
    (goodAlignRect.children()[0] as Latex).opacity(0, 0.8),
    goodAlignEmoji.opacity(1, 0.8),
  );
  yield* waitFor(1);
  yield* all(
    badAlignRect.fill(HIGHLIGHT_COLOR_2, 0.8),
    badAlignRect.stroke(HIGHLIGHT_COLOR_2, 0.8),
    matrix.word1Texts[0].fill(HIGHLIGHT_COLOR_2, 0.8),
    matrix.word2Texts[0].fill(HIGHLIGHT_COLOR_2, 0.8),
    (badAlignRect.children()[0] as Latex).opacity(0, 0.8),
    badAlignEmoji.opacity(1, 0.8),
  );

  yield* waitFor(1);

  // 🎈 Score texts
  const matchScoreTxt = (
    <LetterTxt
      fill={TEXT_FILL}
      fontFamily={TEXT_FONT}
      fontSize={75}
      letterSpacing={4}
      x={300}
      y={70}
    >
      🤩 Match Score
    </LetterTxt>
  ) as LetterTxt;
  const mismatchScoreTxt = (
    <LetterTxt
      fill={TEXT_FILL}
      fontFamily={TEXT_FONT}
      fontSize={75}
      letterSpacing={4}
      x={300}
      y={140}
    >
      😕 Mismatch Score
    </LetterTxt>
  ) as LetterTxt;
  view.add([matchScoreTxt, mismatchScoreTxt]);
  yield* all(
    diagonalStepTxt.y(diagonalStepTxt.y() - 70, 1.0),
    matchScoreTxt.flyIn(1.0, 0.02),
  );
  yield* waitFor(0.3);
  yield* all(
    diagonalStepTxt.y(diagonalStepTxt.y() - 30, 1.0),
    matchScoreTxt.y(matchScoreTxt.y() - 30, 1.0),
    mismatchScoreTxt.flyIn(1.0, 0.02),
  );

  yield* waitFor(1);

  const matchScorePositive = (
    <Latex
      tex="1"
      fill={TEXT_FILL}
      fontSize={70}
      x={910}
      y={30}
      opacity={0}
    />
  ) as Latex;
  const matchScoreNegative = (
    <Latex
      tex="-1"
      fill={TEXT_FILL}
      fontSize={70}
      x={1055}
      y={130}
      opacity={0}
    />
  ) as Latex;
  view.add([matchScorePositive, matchScoreNegative]);
  yield* matchScorePositive.opacity(1, 0.8);
  yield* waitFor(0.5);
  yield* matchScoreNegative.opacity(1, 0.8);

  yield* waitFor(1);

  // 🎈 First mismatch (first diagonal step)

  duration = 0.8;
  yield* all(
    goodAlignRect.fill(null, duration),
    goodAlignRect.stroke(TEXT_FILL, duration),
    badAlignRect.fill(null, duration),
    badAlignRect.stroke(TEXT_FILL, duration),
    goodAlignEmoji.opacity(0, duration),
    badAlignEmoji.opacity(0, duration),
  );

  yield* waitFor(1);

  yield* matrix.step(0, 0, 1, 1, 0.8);

  const highlightMismatchP = createRef<Highlight>();
  const highlightMismatchN = createRef<Highlight>();
  view.add(
    <>
      <Highlight
        ref={highlightMismatchP}
        width={80}
        height={90}
      />
      <Highlight
        ref={highlightMismatchN}
        width={80}
        height={80}
      />
    </>,
  );
  highlightMismatchP().absolutePosition(matrix.word1Texts[0].absolutePosition());
  highlightMismatchP().y(highlightMismatchP().y() + 15);
  highlightMismatchN().absolutePosition(matrix.word2Texts[0].absolutePosition());
  highlightMismatchN().y(highlightMismatchN().y() + 10);
  yield* sequence(0.4,
    highlightMismatchP().highlight(0.8),
    highlightMismatchN().highlight(0.8),
  );

  const firstMismatchCalc = (
    <Latex
      tex="0"
      fill={TEXT_FILL}
      fontSize={80}
      x={530}
      y={-310}
      opacity={0}
    />
  ) as Latex;
  const matchScoreNegativeCopy = matchScoreNegative.snapshotClone();
  view.add([firstMismatchCalc, matchScoreNegativeCopy]);
  yield* all(
    firstMismatchCalc.opacity(1, 0.8),
  );
  yield* waitFor(0.2);
  yield* all(
    matchScoreNegativeCopy.position([620, -310], 1.5),
    matchScoreNegativeCopy.opacity(0, 2.2),
    delay(0.2, firstMismatchCalc.tex("{{0}} {{+}} {{(}} {{-}} {{1}} {{)}}", 1.5)),
  );
  yield* waitFor(0.5);
  yield* all(
    firstMismatchCalc.tex("{{0}} {{-}} {{1}} = -1", 1.8),
  );

  const firstMismatchResult = (
    <Latex
      tex="-1"
      fill={TEXT_FILL}
      fontSize={80}
      x={700}
      y={-310}
      opacity={0}
    />
  ) as Latex;
  view.add(firstMismatchResult);
  matrix.getRectAt(1, 1).children().forEach(child => child.remove()); // remove emoji
  yield* all(
    firstMismatchResult.opacity(1, 0.5),
    firstMismatchResult.position([-535, -320], 1.5),
    firstMismatchResult.fontSize(61, 1.5),
    delay(0.5, firstMismatchResult.fill(TEXT_FILL_DARK, 1.5)),
  );
  yield* all(
    matrix.writeTextAt(1, 1, "-1", 0.0),
    firstMismatchResult.opacity(0, 0.0),
  );

  yield* waitFor(1);

  // 🎈 Second diagonal step (match)

  yield* all(
    firstMismatchCalc.opacity(0, 1.0),
    delay(0.2, all(
      matrix.step(1, 1, 2, 2, 0.8),
    )),
  );

  const highlightMatchY1 = createRef<Highlight>();
  const highlightMatchY2 = createRef<Highlight>();
  view.add(
    <>
      <Highlight
        ref={highlightMatchY1}
        width={80}
        height={90}
      />
      <Highlight
        ref={highlightMatchY2}
        width={80}
        height={80}
      />
    </>,
  );
  highlightMatchY1().absolutePosition(matrix.word1Texts[1].absolutePosition());
  highlightMatchY1().y(highlightMatchY1().y() + 15);
  highlightMatchY2().absolutePosition(matrix.word2Texts[1].absolutePosition());
  highlightMatchY2().y(highlightMatchY2().y() + 15);
  yield* sequence(0.4,
    highlightMatchY1().highlight(0.8),
    highlightMatchY2().highlight(0.8),
  );

  yield* waitFor(0.2);

  const matchCalc = (
    <Latex
      tex="-1"
      fill={TEXT_FILL}
      fontSize={61}
      opacity={0}
    />
  ) as Latex;
  view.add(matchCalc);
  matchCalc.absolutePosition(matrix.getRectAt(1, 1).absolutePosition());

  yield* all(
    matchCalc.opacity(1, 0.7),
    matchCalc.position([540, -310], 1.5),
    matchCalc.fontSize(80, 1.5),
  );

  const matchScoreClone = matchScorePositive.snapshotClone();
  view.add(matchScoreClone);

  yield* all(
    matchScoreClone.position([660, -315], 1.5),
    delay(0.26, chain(
      matchCalc.tex("{{-1}} {{+}} {{1}}", 1.5),
      matchCalc.tex("{{-1}} {{+}} {{1}} {{=}} {{0}}", 1.5),
    )),
    delay(0.45, matchScoreClone.opacity(0, 1.5)),
  );

  yield* waitFor(0.2);

  yield* all(
    matchCalc.tex("0", 1.0),
    delay(0.5, matchCalc.fill(TEXT_FILL_DARK, 1.5)),
    matchCalc.fontSize(61, 1.5),
    matchCalc.absolutePosition(matrix.getRectAt(2, 2).absolutePosition(), 1.5),
  );
  matchCalc.remove();
  matrix.getRectAt(2, 2).children().forEach(child => child.remove());
  yield* matrix.writeTextAt(2, 2, "0", 0.0);
  const secondDiagRect = matrix.getRectAt(2, 2);
  yield* all(
    secondDiagRect.fill(null, 0.8),
    (secondDiagRect.children()[0] as Txt).fill(TEXT_FILL, 0.8),
  );

  yield* waitFor(1);

  // 🎈 First field from multiple directions

  const oneDiagRect = matrix.getRectAt(1, 1);
  yield* all(
    matrix.step(0, 0, 1, 1, 0.8),
    delay(0.35, (oneDiagRect.children()[0] as Txt).fill(TEXT_FILL_DARK, 0.8)),
  );

  yield* waitFor(5);
});
