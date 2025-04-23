import { Line, makeScene2D, Node, Rect } from "@motion-canvas/2d";
import { all, createRef, delay, sequence, Spring, spring, waitFor } from "@motion-canvas/core";
import { AlignState } from "./AlignState";
import { TEXT_FILL } from "./globals";
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
          ...matrix.word1Texts.map((text, i) => text.opacity(1, 1.5)),
          ...matrix.word2Texts.map((text, i) => text.opacity(1, 1.5)),
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
      x={1000}
    />,
  );

  const scoreRuler = (
    <ScoreRuler
      points={[
        [0, 400],
        [0, -400],
      ]}
      x={660}
      maxValue={30}
    />
  ) as ScoreRuler;
  view.add(scoreRuler);
  scoreRuler.ruler.end(0);
  scoreRuler.value(-30);

  yield* all(
    sequence(0.06,
      alignmentContainer().x(-1100, 1.5),
      toMatrixLine().x(-380, 1.5),
      matrixContainer().x(300, 1.5),
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
      scoreRuler.value(20, 1.4),
    )),
  );

  yield* scoreRuler.value(-7, 1.2);

  yield* waitFor(5);
});
