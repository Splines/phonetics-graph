import { is, Latex, Line, makeScene2D, Node, Rect, Txt } from "@motion-canvas/2d";
import { all, createRef, delay, sequence, Spring, spring, waitFor } from "@motion-canvas/core";
import { AlignState } from "./AlignState";
import { HIGHLIGHT_COLOR, TEXT_FILL, TEXT_FILL_DARK, TEXT_FONT } from "./globals";
import { Highlight } from "./Highlight";
import { LetterTxt } from "./LetterTxt";
import { Matrix } from "./Matrix";
import { ScoreRuler } from "./ScoreRuler";

export default makeScene2D(function* (view) {
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

  // 🎈 Diagonal steps

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

  yield* waitFor(5);
});
