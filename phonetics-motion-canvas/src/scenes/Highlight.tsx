import { Rect, RectProps } from "@motion-canvas/2d";
import { all, chain, ThreadGenerator, waitFor } from "@motion-canvas/core";
import { HIGHLIGHT_COLOR } from "./globals";

export class Highlight extends Rect {
  private highlightRect: Rect;

  public constructor({ children, ...props }: RectProps) {
    super(props);
    if (children) throw new Error("A highlight cannot have children");

    const highlightRect = (
      <Rect
        width={200}
        height={100}
        stroke={HIGHLIGHT_COLOR}
        lineWidth={5}
        {...props}
        opacity={0}
        end={0}
      >
      </Rect>
    ) as Rect;

    this.add(highlightRect);
    this.highlightRect = highlightRect;
  }

  * highlight(duration: number): ThreadGenerator {
    this.reset();

    yield* all(
      this.highlightRect.opacity(1, 0.2 * duration),
      this.highlightRect.end(1, duration),
      chain(
        waitFor(0.3 * duration),
        all(
          this.highlightRect.start(1, duration),
        ),
      ),
    );

    this.reset();
  }

  public rect(): Rect {
    return this.highlightRect;
  }

  private reset() {
    this.highlightRect.opacity(0);
    this.highlightRect.start(0);
    this.highlightRect.end(0);
  }
}
