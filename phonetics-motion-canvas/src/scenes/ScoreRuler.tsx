import { Line, LineProps } from "@motion-canvas/2d";
import { createSignal, SimpleSignal } from "@motion-canvas/core";

export class ScoreRuler extends Line {
  private ruler: Line;
  private value: SimpleSignal<number> = createSignal(0);

  public constructor({ children, ...props }: LineProps) {
    super(props);
    if (children) throw new Error("A score ruler cannot have children");
    if (!props.points) {
      throw new Error("A score ruler must have line points");
    }

    const ruler = (
      <Line
        points={props.points}
        stroke="white"
        lineWidth={12}
        lineCap="round"
        {...props}
        endArrow
        arrowSize={25}
      />
    ) as Line;

    this.add(ruler);
    this.ruler = ruler;
  }
}
