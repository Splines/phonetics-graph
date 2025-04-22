import { makeScene2D, Node, Rect } from "@motion-canvas/2d";
import { createRef, waitFor } from "@motion-canvas/core";
import { AlignState } from "./AlignState";

export default makeScene2D(function* (view) {
  const alignmentContainer = createRef<Node>();
  const alignment = new AlignState(alignmentContainer, "pɥisɑ̃s", "nɥɑ̃s", "..--..");
  const texts = alignment.generateElements();

  view.add(
    <Rect ref={alignmentContainer}>
      { texts }
    </Rect>,
  );

  yield* alignment.animateToState(":-.:-:---", 1.2);
  yield* waitFor(5);
});
