import { is, Latex, Line, makeScene2D, Node, Rect, Txt } from "@motion-canvas/2d";
import {
  all,
  chain,
  createRef, delay,
  sequence, Spring,
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
  let newRectWidth = 150;
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

  yield* waitFor(0.5);

  // 🎈 First field from multiple directions

  // shift entire matrix
  newRectWidth = 150;
  duration = 1.0;
  const dontHideList = [2, 7, 8, 12, 13, 14];
  yield* all(
    // remove old text
    diagonalStepTxt.opacity(0, 1.0),
    matchScoreTxt.opacity(0, duration),
    mismatchScoreTxt.opacity(0, duration),
    matchScorePositive.opacity(0, duration),
    matchScoreNegative.opacity(0, duration),
    // increase matrix gaps & move
    matrix.layout().width(1900, 1.5),
    matrix.layout().gap(180, 1.5),
    matrixContainer().y(666, 1.5),
    matrixContainer().x(380, 1.5),
    // hide rest of matrix
    ...matrix.layout().children().map((child, i) => {
      if (dontHideList.includes(i)) {
        return null;
      }
      return child.opacity(0.0, 1.5);
    }),
  );

  const oneDiagRect = matrix.getRectAt(1, 1);
  yield* (oneDiagRect.children()[0] as Txt).opacity(0, 1.0);

  yield* waitFor(1);

  yield* matrix.step(0, 0, 1, 1, 1.0);
  yield* waitFor(0.5);
  yield* all(
    oneDiagRect.fill(null, 0.8),
    delay(0.25, matrix.step(0, 1, 1, 1, 0.8)),
  );
  yield* waitFor(0.5);
  yield* all(
    oneDiagRect.fill(null, 0.8),
    delay(0.25, matrix.step(1, 0, 1, 1, 0.8)),
  );

  yield* waitFor(0.5);

  // 🎈 Trinity of diag, up & left
  const arrowListOffset = 200;

  // Diagonal
  const [arrow1Anim, arrow1] = matrix.stepAndArrowStay(0, 0, 1, 1, 1.0);
  yield* arrow1Anim;
  const diagArrow = arrow1.snapshotClone();
  view.add(diagArrow);
  diagArrow.absolutePosition(arrow1.absolutePosition());
  const textProps = {
    fill: TEXT_FILL,
    fontSize: 80,
    opacity: 0,
  };
  const diagCalc = (
    <Latex
      tex="0 - 1 = {{-1}}"
      {...textProps}
      x={() => diagArrow.x()}
    />
  ) as Latex;
  view.add(diagCalc);

  yield* all(
    matrixContainer().x(matrixContainer().x() - 500, 1.5),
    diagArrow.x(700, 1.5),
    delay(0.35, diagCalc.opacity(1, 1.5)),
  );

  const highlight1 = (
    <Highlight
      width={140}
      height={80}
      x={435}
      y={0}
    />
  ) as Highlight;
  view.add(highlight1);
  yield* highlight1.highlight(0.8);

  yield* waitFor(1);

  // From top to bottom
  const [arrow2Anim, arrow2] = matrix.stepAndArrowStay(0, 1, 1, 1, 1.0);
  yield* arrow2Anim;
  yield* waitFor(0.5);
  const fromAboveArrow = arrow2.snapshotClone();
  view.add(fromAboveArrow);
  fromAboveArrow.absolutePosition(arrow2.absolutePosition());
  const fromAboveCalc = (
    <Latex
      tex="-2 - 2 = {{-4}}"
      {...textProps}
      x={() => fromAboveArrow.x() + 200}
    />
  ) as Latex;
  view.add(fromAboveCalc);

  yield* all(
    diagArrow.y(diagArrow.y() - arrowListOffset, 1.5),
    diagCalc.y(diagCalc.y() - arrowListOffset, 1.5),
    fromAboveArrow.x(530, 1.5),
    delay(0.35, fromAboveCalc.opacity(1, 1.5)),
  );

  const highlight2 = (
    <Highlight
      width={140}
      height={80}
      x={471}
      y={0}
    />
  ) as Highlight;
  view.add(highlight2);
  yield* highlight2.highlight(0.8);

  yield* waitFor(1);

  // From left to right
  const [arrow3Anim, arrow3] = matrix.stepAndArrowStay(1, 0, 1, 1, 1.0);
  yield* arrow3Anim;
  const fromLeftArrow = arrow3.snapshotClone();
  view.add(fromLeftArrow);
  fromLeftArrow.absolutePosition(arrow3.absolutePosition());
  const fromLeftCalc = (
    <Latex
      tex="-2 - 2 = {{-4}}"
      {...textProps}
      x={() => fromLeftArrow.x() + 30}
      y={165}
    />
  ) as Latex;
  view.add(fromLeftCalc);

  yield* all(
    fromLeftArrow.x(700, 1.5),
    delay(0.35, fromLeftCalc.opacity(1, 1.5)),
  );

  const highlight3 = (
    <Highlight
      width={140}
      height={80}
      x={471}
      y={81}
    />
  ) as Highlight;
  view.add(highlight3);
  yield* highlight3.highlight(0.8);

  yield* waitFor(1);

  // 🎈 What value to keep? -> Maximum

  yield* all(
    diagCalc.tex("-1", 1.5),
    diagCalc.x(diagCalc.x() - 220, 1.5),
    fromAboveCalc.tex("-4", 1.5),
    fromAboveCalc.x(fromAboveCalc.x() - 250, 1.5),
    fromLeftCalc.tex("-4", 1.5),
    fromLeftCalc.x(fromLeftCalc.x() - 250, 1.5),
  );

  yield* waitFor(1);

  const maxLatex = (
    <Latex
      tex="{{ \text{max} }} {{ \bigl\{ }} {{ \quad\quad,\quad\quad,\quad\quad }} {{ \bigr\} }}"
      fill={TEXT_FILL}
      fontSize={80}
      x={450}
      y={0}
      opacity={0}
    />
  ) as Latex;
  view.add(maxLatex);

  yield* all(
    matrixContainer().x(matrixContainer().x() - 100, 1.5),

    diagArrow.position([700, 820], 1.5),
    fromAboveArrow.position([740, 820], 1.5),
    fromLeftArrow.position([1150, 650], 1.5),

    diagCalc.position([300, 0], 1.5),
    fromAboveCalc.position([530, 0], 1.5),
    fromLeftCalc.position([760, 0], 1.5),

    delay(1.0, maxLatex.opacity(1, 1.5)),
  );

  const maxLatexResult = (
    <Latex
      tex="{{=}} {{-1}}"
      fill={TEXT_FILL}
      fontSize={80}
      opacity={0}
      x={950}
    />
  ) as Latex;
  view.add(maxLatexResult);

  const highlightMaxResult = (
    <Highlight
      width={140}
      height={80}
      x={555}
    />
  ) as Highlight;
  view.add(highlightMaxResult);

  yield* all(
    maxLatexResult.x(1060, 0.9),
    maxLatexResult.opacity(1, 0.9),
    delay(0.35, highlightMaxResult.highlight(0.8)),
  );

  yield* waitFor(1);

  const resultMin1 = (
    <Latex
      tex="-1"
      fill={TEXT_FILL}
      fontSize={80}
      x={1111}
    />
  ) as Latex;
  view.add(resultMin1);

  yield* all(
    oneDiagRect.fill(null, 1.5),
    resultMin1.fontSize(61, 1.5),
    resultMin1.position([-435, 171], 1.5),
  );
  resultMin1.remove();
  oneDiagRect.children().forEach(child => child.opacity(1));

  yield* waitFor(1);

  // 🎈 user fields: -3 both cases

  const fadeOutMaxText = all(
    diagCalc.opacity(0, 1.5),
    fromAboveCalc.opacity(0, 1.5),
    fromLeftCalc.opacity(0, 1.5),
    diagArrow.start(1, 1.5),
    diagArrow.opacity(0, 1.5),
    fromAboveArrow.start(1, 1.5),
    fromAboveArrow.opacity(0, 1.5),
    fromLeftArrow.start(1, 1.5),
    fromLeftArrow.opacity(0, 1.5),
    maxLatex.opacity(0, 1.5),
    maxLatexResult.opacity(0, 1.5),
  );

  const fadeOutArrows = all(
    arrow1.opacity(0, 1.5),
    arrow1.start(1, 1.5),
    arrow2.opacity(0, 1.5),
    arrow2.start(1, 1.5),
    arrow3.opacity(0, 1.5),
    arrow3.start(1, 1.5),
  );

  const showAgainIndices = [3, 9, 15, 18, 19, 20];
  const showMoreFieldsAgain = matrix.layout().children().map((child, i) => {
    if (!showAgainIndices.includes(i)) {
      return null;
    }
    return child.opacity(1.0, 1.8);
  });

  const userFieldHighlight1 = createRef<Highlight>();
  const userFieldHighlight2 = createRef<Highlight>();

  view.add(
    <>
      <Highlight
        ref={userFieldHighlight1}
        width={190}
        height={190}
        x={165}
        y={35}
      />
      <Highlight
        ref={userFieldHighlight2}
        width={180}
        height={180}
        x={0}
        y={200}
      />
    </>,
  );

  yield* all(
    matrixContainer().x(matrixContainer().x() + 435, 2.5),
    matrixContainer().y(matrixContainer().y() - 100, 2.5),
    fadeOutMaxText,
    fadeOutArrows,
    ...showMoreFieldsAgain,
    delay(0.2, all(
      userFieldHighlight1().highlight(1.3),
      delay(0.2, userFieldHighlight2().highlight(1.3)),
    )),
  );
  yield* waitFor(1); // give user time

  yield* all(
    matrix.writeTextAt(1, 2, "-3", 1.0),
    matrix.writeTextAt(2, 1, "-3", 1.0),
  );

  yield* waitFor(1);

  const examineRect = matrix.getRectAt(2, 2);
  examineRect.children().forEach(child => child.remove());
  examineRect.stroke(matrix.BASE_COLOR);

  const highlightExamine = (
    <Highlight
      width={180}
      height={180}
      x={165}
      y={200}
    />
  ) as Highlight;
  view.add(highlightExamine);

  yield* all(
    matrix.getRectAt(1, 2).fill(null, 1.0),
    (matrix.getRectAt(1, 2).children()[0] as Txt).fill(TEXT_FILL, 1.0),
    matrix.getRectAt(2, 1).fill(null, 1.0),
    (matrix.getRectAt(2, 1).children()[0] as Txt).fill(TEXT_FILL, 1.0),

    delay(0.5, examineRect.opacity(1, 1.0)),
    delay(0.7, highlightExamine.highlight(1.0)),
  );

  yield* waitFor(1.5);

  // 🎈 Examine rect again together

  // diagonal
  const [examineAgain, examineDiag] = matrix.stepAndArrowStay(1, 1, 2, 2, 0.8);
  const diagPlusOne = (
    <Latex
      tex="+1"
      fill={HIGHLIGHT_COLOR}
      fontSize={65}
      opacity={0}
    />
  ) as Latex;
  view.add(diagPlusOne);
  diagPlusOne.absolutePosition(examineDiag.absolutePosition());
  diagPlusOne.y(diagPlusOne.y() - 420);
  yield* all(
    examineAgain,
    delay(0.2, all(
      diagPlusOne.opacity(1, 0.8),
      diagPlusOne.y(diagPlusOne.y() + 50, 0.8),
    )),
  );
  yield* matrix.writeTextAt(2, 2, "0", 1.2);
  yield* waitFor(0.5);

  const examineMax = (
    <Latex
      tex="{{ \text{max} }} {{ \bigl\{ }} {{ 0 }} {{ \bigr\} }}"
      fill={TEXT_FILL}
      fontSize={75}
      x={350}
      y={400}
      opacity={0}
    />
  ) as Latex;
  view.add(examineMax);

  yield* all(
    examineMax.opacity(1, 1.3),
    examineMax.x(examineMax.x() + 300, 1.3),
    examineDiag.start(1, 1.3),
    examineDiag.opacity(0, 1.3),
    diagPlusOne.opacity(0, 1.3),
    examineRect.fill(null, 1.3),
    (examineRect.children()[0] as Latex).opacity(0, 1.3),
  );
  examineRect.children().forEach(child => child.remove());
  yield* waitFor(1.0);

  // top to bottom
  const [examineFromTopAnim, examineFromTop] = matrix.stepAndArrowStay(1, 2, 2, 2, 0.8);
  const fromTop = (
    <Latex
      tex="-2"
      fill={HIGHLIGHT_COLOR}
      fontSize={65}
      opacity={0}
    />
  ) as Latex;
  view.add(fromTop);
  fromTop.absolutePosition(examineFromTop.absolutePosition());
  fromTop.y(fromTop.y() - 385);
  fromTop.x(fromTop.x() + 10);
  yield* all(
    examineFromTopAnim,
    delay(0.2, all(
      fromTop.opacity(1, 0.8),
      fromTop.y(fromTop.y() + 50, 0.8),
    )),
  );
  yield* matrix.writeTextAt(2, 2, "-5", 1.2);
  yield* waitFor(0.5);

  yield* all(
    examineMax.tex("{{ \\text{max} }} {{ \\bigl\\{ }} {{ 0 }}, {{-5}} {{ \\bigr\\} }}", 1.3),
    examineMax.x(examineMax.x() + 80, 1.3),
    examineFromTop.start(1, 1.3),
    examineFromTop.opacity(0, 1.3),
    fromTop.opacity(0, 1.3),
    examineRect.fill(null, 1.3),
    (examineRect.children()[0] as Latex).opacity(0, 1.3),
  );
  examineRect.children().forEach(child => child.remove());
  yield* waitFor(1.0);

  // left to right
  const [examineFromLeftAnim, examineFromLeft] = matrix.stepAndArrowStay(2, 1, 2, 2, 0.8);
  const fromLeft = (
    <Latex
      tex="-2"
      fill={HIGHLIGHT_COLOR}
      fontSize={65}
      opacity={0}
    />
  ) as Latex;
  view.add(fromLeft);
  fromLeft.absolutePosition(examineFromLeft.absolutePosition());
  fromLeft.y(fromLeft.y() - 80);
  fromLeft.x(fromLeft.x() - 100);
  yield* all(
    examineFromLeftAnim,
    delay(0.2, all(
      fromLeft.opacity(1, 0.8),
      fromLeft.x(fromLeft.x() + 50, 0.8),
    )),
  );
  yield* matrix.writeTextAt(2, 2, "-5", 1.2);
  yield* waitFor(0.5);

  yield* all(
    // eslint-disable-next-line @stylistic/max-len
    examineMax.tex("{{ \\text{max} }} {{ \\bigl\\{ }} {{ 0 }}, {{-5}}, {{-5}} {{ \\bigr\\} }}", 1.3),
    examineMax.x(examineMax.x() + 80, 1.3),
    examineFromLeft.start(1, 1.3),
    examineFromLeft.opacity(0, 1.3),
    fromLeft.opacity(0, 1.3),
    examineRect.fill(null, 1.3),
    (examineRect.children()[0] as Latex).opacity(0, 1.3),
  );
  examineRect.children().forEach(child => child.remove());
  yield* waitFor(1.0);

  // finally
  yield* all(
    examineMax.x(examineMax.x() - 150, 1.3),
    examineMax.opacity(0, 1.3),
    delay(0.2, matrix.writeTextAt(2, 2, "0", 1.3)),
  );
  yield* waitFor(1);

  // 🎈 Entire matrix
  duration = 1.9;
  yield* all(
    matrix.layout().width(1100, 1.5 * duration),
    matrix.layout().gap(20, 1.5 * duration),
    matrixContainer().x(0, 1.3 * duration),
    matrixContainer().y(0, 1.3 * duration),
    delay(0.2,
      sequence(0.06,
        ...matrix.layout().children().map((child) => {
          return child.opacity(1.0, 0.9);
        },
        ),
      ),
    ),
    delay(0.7, all(
      examineRect.fill(null, 1.7),
      (examineRect.children()[0] as Latex).fill(TEXT_FILL, 1.7),
    )),
  );
  yield* waitFor(1.5); // 3,2,1 countdown

  duration = 1.2;
  const matrixReveal = [
    matrix.writeTextAt(1, 3, "-5", duration, false),
    matrix.writeTextAt(1, 4, "-7", duration, false),
    matrix.writeTextAt(2, 3, "-2", duration, false),
    matrix.writeTextAt(2, 4, "-4", duration, false),
    matrix.writeTextAt(3, 1, "-5", duration, false),
    matrix.writeTextAt(3, 2, "-2", duration, false),
    matrix.writeTextAt(3, 3, "-1", duration, false),
    matrix.writeTextAt(3, 4, "-3", duration, false),
    matrix.writeTextAt(4, 1, "-7", duration, false),
    matrix.writeTextAt(4, 2, "-4", duration, false),
    matrix.writeTextAt(4, 3, "-3", duration, false),
    matrix.writeTextAt(4, 4, "0", duration, false),
    matrix.writeTextAt(5, 1, "-9", duration, false),
    matrix.writeTextAt(5, 2, "-6", duration, false),
    matrix.writeTextAt(5, 3, "-3", duration, false),
    matrix.writeTextAt(5, 4, "-2", duration, false),
    matrix.writeTextAt(6, 1, "-11", duration, false),
    matrix.writeTextAt(6, 2, "-8", duration, false),
    matrix.writeTextAt(6, 3, "-5", duration, false),
    matrix.writeTextAt(6, 4, "-2", duration, false),
  ];
  const desaturate = matrix.layout()
    .children().map(child => (child as Txt).stroke(matrix.BASE_COLOR, duration));
  yield* all(
    sequence(0.06, ...matrixReveal),
    // delay(0.5, sequence(0.04, ...desaturate)),
  );
  yield waitFor(1);

  // 🎈 Summary Texts
  const textPropsSummary = {
    fill: TEXT_FILL,
    fontFamily: TEXT_FONT,
    fontSize: 80,
    letterSpacing: 4,
  };
  const textPropsSummaryLatex = {
    fill: TEXT_FILL,
    fontSize: 72,
    opacity: 0,
  };

  const summaryGapPenalty = (
    <LetterTxt
      {...textPropsSummary}
      x={120}
      y={0}
    >
      Gap Penalty
    </LetterTxt>
  ) as LetterTxt;
  const summaryGapLatex = (
    <Latex
      tex="p = -2"
      {...textPropsSummaryLatex}
      x={715}
      y={0}
    />
  ) as Latex;
  view.add([summaryGapPenalty, summaryGapLatex]);
  yield* all(
    matrixContainer().x(matrixContainer().x() - 500, 1.8),
    delay(0.5, all(
      summaryGapPenalty.flyIn(1.0, 0.02),
      delay(0.5, summaryGapLatex.opacity(1, 1.0)),
    )),
  );
  yield* waitFor(0.5);

  const summaryMatchScore = (
    <LetterTxt
      {...textPropsSummary}
      x={120}
      y={20}
    >
      🤩 Match Score
    </LetterTxt>
  ) as LetterTxt;
  const summaryMatchLatex = (
    <Latex
      tex="1"
      {...textPropsSummaryLatex}
      x={770}
      y={12}
    />
  ) as Latex;
  view.add([summaryMatchScore, summaryMatchLatex]);
  yield* all(
    summaryGapPenalty.y(summaryGapPenalty.y() - 110, 1.0),
    summaryGapLatex.y(summaryGapLatex.y() - 110, 1.0),
    summaryMatchScore.flyIn(1.0, 0.02),
    delay(0.5, summaryMatchLatex.opacity(1, 1.0)),
  );
  yield* waitFor(0.5);

  const summaryMismatchScore = (
    <LetterTxt
      {...textPropsSummary}
      x={120}
      y={140}
    >
      😕 Mismatch Score
    </LetterTxt>
  ) as LetterTxt;
  const summaryMismatchLatex = (
    <Latex
      tex="-1"
      {...textPropsSummaryLatex}
      x={915}
      y={135}
    />
  ) as Latex;
  view.add([summaryMismatchScore, summaryMismatchLatex]);
  yield* all(
    summaryMismatchScore.flyIn(1.0, 0.02),
    delay(0.5, summaryMismatchLatex.opacity(1, 1.0)),
  );
  yield* waitFor(1.5);
});
