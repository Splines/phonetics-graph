import { initial, Line, LineProps, signal } from "@motion-canvas/2d";
import { createSignal, SignalValue, SimpleSignal } from "@motion-canvas/core";

export interface ScoreRulerProps extends LineProps {
  maxValue: SignalValue<number>;
}

export class ScoreRuler extends Line {
  @initial(0)
  @signal()
  public declare readonly maxValue: SimpleSignal<number, this>;

  private ruler: Line;
  public value: SimpleSignal<number> = createSignal(0); // range: [-100, 100]

  public constructor({ children, ...props }: ScoreRulerProps) {
    super(props);

    if (children) throw new Error("A score ruler cannot have children");
    if (!props.points) {
      throw new Error("A score ruler must have line points");
    }
    if (props.points.length !== 2) {
      throw new Error("A score ruler must have exactly 2 points");
    }
    if (props.points[0][0] !== props.points[1][0]) {
      throw new Error("A score ruler must be vertical");
    }

    const lineHeight = Math.abs(props.points[1][1] - props.points[0][1]) - 120;

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

    // eslint-disable-next-line no-unused-vars
    const { points, ...propsWithoutPoints } = props;
    const valueMarker = (
      <Line
        points={[[-25, 0], [25, 0]]}
        lineWidth={12}
        stroke="white"
        lineCap="round"
        {...propsWithoutPoints}
      />
    ) as Line;
    this.add(valueMarker);

    valueMarker.y(() => -(this.value() / this.maxValue()) * (lineHeight / 2));
  }
}
