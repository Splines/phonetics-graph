import { Node, Rect, Txt, TxtProps } from "@motion-canvas/2d";
import {
  all,
  easeInOutQuart, sequence, ThreadGenerator, TimingFunction,
} from "@motion-canvas/core";

export class LetterTxt extends Node {
  private Y_OFFSET = 80;
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
    ctx.letterSpacing = `${props.letterSpacing ?? 7}px`;

    let cursorX = 0;
    let previousWidth = 0;

    for (let i = 0; i < text.length; i++) {
      const char = text[i];
      let txtChar = char;

      // Surrogate pairs for emojis 🤩
      if (i < text.length - 1 && char >= "\uD800" && char <= "\uDBFF") {
        const nextChar = text[i + 1];
        if (nextChar >= "\uDC00" && nextChar <= "\uDFFF") {
          txtChar = char + nextChar;
        }
      }

      const currentWidth = ctx.measureText(text.substring(0, i + 1)).width;
      const charWidth = currentWidth - previousWidth;
      previousWidth = currentWidth;

      const txt = this.constructTxt(txtChar, cursorX, props);
      container.add(txt);
      this.txtObjects.push(txt);

      cursorX += charWidth;

      if (txtChar.length > 1) {
        i += 1; // skip the next character as it's part of the surrogate pair
      }
    }
  }

  private constructTxt(text: string, cursorX: number, props: any): Txt {
    return (
      <Txt
        {...props}
        x={cursorX}
        y={this.Y_OFFSET}
        opacity={0}
        offsetX={-1}
      >
        {text}
      </Txt>
    ) as Txt;
  }

  /**
   * Animates the text using a smooth fly-in effect.
   */
  * flyIn(duration: number, delay: number,
            easing: TimingFunction = easeInOutQuart): ThreadGenerator {
    const generators: ThreadGenerator[] = [];

    for (const txt of this.txtObjects) {
      generators.push(all(
        txt.opacity(1, duration),
        txt.position.y(0, duration, easing),
      ));
    }

    yield* sequence(delay, ...generators);
  }
}
