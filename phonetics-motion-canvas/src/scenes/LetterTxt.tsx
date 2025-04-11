import { Node, Rect, Txt, TxtProps } from "@motion-canvas/2d";
import { all, sequence, ThreadGenerator } from "@motion-canvas/core";

export interface LetterTxtProps extends TxtProps {}

export class LetterTxt extends Node {
  private txtObjects: Txt[] = [];

  public constructor({ children, ...props }: TxtProps) {
    super(props);
    if (!children) throw new Error("Text is required");

    const text = children.toString();
    const container = <Rect />;
    this.add(container);

    // Create offscreen canvas for text measurement
    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d")!;
    // Configure font to match what Txt will use
    const fontSize = props.fontSize ?? 40;
    const fontFamily = props.fontFamily ?? "sans-serif";
    const fontWeight = props.fontWeight ?? "normal";
    const fontStyle = props.fontStyle ?? "normal";
    ctx.font = `${fontStyle} ${fontWeight} ${fontSize}px ${fontFamily}`;
    ctx.textBaseline = "bottom";
    if ("letterSpacing" in ctx) {
      ctx.letterSpacing = `${props.letterSpacing}px`;
    }

    let cursorX = 0;

    for (const char of text) {
      const txt = (
        <Txt
          fontFamily={props.fontFamily}
          fontSize={props.fontSize}
          fontWeight={props.fontWeight}
          fontStyle={props.fontStyle}
          fill={props.fill}
          letterSpacing={0}
          x={cursorX}
          y={80}
          opacity={0}
        >
          {char}
        </Txt>
      ) as Txt;

      container.add(txt);
      this.txtObjects.push(txt);

      const metrics = ctx.measureText(char);
      cursorX += metrics.width;
    }
  }

  * flyIn(duration: number, delay: number): ThreadGenerator {
    const generators: ThreadGenerator[] = [];

    for (const txt of this.txtObjects) {
      generators.push(all(
        txt.opacity(1, duration),
        txt.position.y(0, duration),
      ));
    }

    yield* sequence(delay, ...generators);
  }
}
