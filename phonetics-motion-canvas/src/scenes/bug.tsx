import { Line, makeScene2D, Node } from "@motion-canvas/2d";
import { Vector2, waitFor } from "@motion-canvas/core";

/* export default makeScene2D(function* (view) {
  const container = createRef<Node>();
  view.add(<Node ref={container} x={-600} />);

  const rect = (
    <Rect
      width={100}
      height={100}
      x={0}
      y={0}
      stroke="white"
      lineWidth={10}
    />) as Rect;
  container().add(rect);

  yield* container().x(-500, 1.0);
  yield* waitFor(1.0);
  yield* rect.position([200, 200], 1.0);

  const rectClone = rect.snapshotClone() as Rect;
  view.add(rectClone);
  rectClone.stroke("yellow");
  rectClone.absolutePosition(rect.absolutePosition());
  rectClone.position([0, 0]);

  yield* waitFor(2.0);
}); */

// export default makeScene2D(function* (view) {
//   const container = createRef<Node>();
//   view.add(<Node ref={container} x={-600} />);

//   const arrow = (
//     <Line
//       points={[
//         [40, 50],
//         [60, -50],
//       ]}
//       lineWidth={10}
//       stroke="white"
//       endArrow
//       lineCap="round"
//     />
//   ) as Line;
//   container().add(arrow);

//   yield* container().x(-500, 1.0);
//   yield* waitFor(0.5);
//   yield* arrow.position([200, 200], 1.0);

//   const arrowClone = arrow.snapshotClone();
//   arrowClone.stroke("yellow");
//   view.add(arrowClone);
//   yield* waitFor(1.0);
//   yield* arrowClone.position([0, 0], 1.0);

//   yield* waitFor(2.0);
// });

export default makeScene2D(function* (view) {
  const arrow = (
    <Line
      points={[
        [40, 50],
        [60, -50],
      ]}
      lineWidth={10}
      stroke="white"
      endArrow
    />
  ) as Line;
  view.add(arrow);

  const arrowClone = arrow.snapshotClone();
  view.add(arrowClone);
  arrowClone.stroke("yellow");
  arrowClone.position([0, 0]);
  yield* waitFor(2);

  yield* arrowClone.position(positionAtCenterOfMass(arrowClone, 0, 0), 1.0);
  yield* waitFor(2);
});

function positionAtCenterOfMass(node: Node, x: number, y: number): Vector2 {
  const bounds = node.cacheBBox();
  const centerX = (bounds.left + bounds.right) / 2;
  const centerY = (bounds.top + bounds.bottom) / 2;
  return new Vector2(x - centerX, y - centerY);
}

/**
 * Moves the node so that its center of mass (center of local bounds) is at
 * the origin.
 */
function moveNodeCenterOfMassToOrigin(node: Node) {
  const bounds = node.cacheBBox();
  const centerX = (bounds.left + bounds.right) / 2;
  const centerY = (bounds.top + bounds.bottom) / 2;
  node.position([-centerX, -centerY]);
}
