import { initial, Line, LineProps, signal, Txt } from "@motion-canvas/2d";
import { createSignal, SignalValue, SimpleSignal, unwrap } from "@motion-canvas/core";
import { HIGHLIGHT_COLOR, TEXT_FONT } from "./globals";

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
    const getY = (value: number) => {
      return -(value / this.maxValue()) * (lineHeight / 2);
    };

    // 🌟 Ruler
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

    // 🌟 Grid lines
    // eslint-disable-next-line no-unused-vars
    const { points, ...propsWithoutPoints } = props;

    const step = 20;
    for (let i = -1; i <= 1; i++) {
      const neutralLine = (
        <Line
          points={[[-17, getY(i * step)], [17, getY(i * step)]]}
          lineWidth={10}
          stroke="white"
          lineCap="round"
          {...propsWithoutPoints}
        />
      ) as Line;
      this.add(neutralLine);
    }

    // 🌟 Value Marker
    const valueMarker = (
      <Line
        points={[[-27, 0], [27, 0]]}
        lineWidth={15}
        stroke={HIGHLIGHT_COLOR}
        lineCap="round"
        {...propsWithoutPoints}
      />
    ) as Line;
    this.add(valueMarker);
    valueMarker.y(() => getY(this.value()));

    // 🌟 Value Text
    const valueText = (
      <Txt
        text={() => this.value().toFixed(1)}
        fill={HIGHLIGHT_COLOR}
        fontFamily={TEXT_FONT}
        fontWeight={700}
        fontSize={80}
        x={(unwrap(props.x) ?? 0) + 130}
        y={() => getY(this.value()) + 4}
      />
    ) as Txt;
    this.add(valueText);

    this.opacity(0);
  }
}
