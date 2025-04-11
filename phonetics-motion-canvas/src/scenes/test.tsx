import { makeScene2D, Rect } from "@motion-canvas/2d";
import { createRef, waitFor } from "@motion-canvas/core";

export default makeScene2D(function* (view) {
  const rect = createRef<Rect>();
  const highlightColor = "#FFEB6C";

  view.add(
    <Rect
      ref={rect}
      rotation={180}
      lineWidth={5}
      stroke={highlightColor}
      width={100}
      height={200}
      radius={8}
      closed
      end={0}
    />,
  );

  yield* waitFor(0.5);
  yield* rect().end(1, 0.8);
  yield* waitFor(0.5);
});
