import { initial, Line, LineProps, signal, Txt } from "@motion-canvas/2d";
import { createSignal, SignalValue, SimpleSignal, unwrap } from "@motion-canvas/core";
import { TEXT_FONT } from "./globals";

export interface ScoreRulerProps extends LineProps {
  maxValue: SignalValue<number>;
}

export class ScoreRuler extends Line {
  @initial(0)
  @signal()
  public declare readonly maxValue: SimpleSignal<number, this>;

  public ruler: Line;
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
    const myPoints = unwrap(props.points);
    const startPoint = myPoints[0].valueOf() as [number, number];
    const endPoint = myPoints[1].valueOf() as [number, number];

    if (startPoint[0] !== endPoint[0]) {
      throw new Error("A score ruler must be vertical");
    }

    const lineHeight = Math.abs(endPoint[1] - startPoint[1]) - 120;

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

    const getY = (value: number) => {
      return -(value / this.maxValue()) * (lineHeight / 2);
    };

    valueMarker.y(() => getY(this.value()));

    const valueText = (
      <Txt
        text={() => this.value().toFixed(0)}
        fill="white"
        fontFamily={TEXT_FONT}
        fontWeight={700}
        fontSize={80}
        x={(unwrap(props.x) ?? 0) + 100}
        y={() => getY(this.value()) + 4}
      />
    ) as Txt;
    this.add(valueText);

    this.opacity(0);
  }
}
