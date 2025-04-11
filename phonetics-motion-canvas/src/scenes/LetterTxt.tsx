import { Node, Rect, Txt, TxtProps } from "@motion-canvas/2d";
import { all, sequence, ThreadGenerator } from "@motion-canvas/core";

export interface LetterTxtProps extends TxtProps {
}

export class LetterTxt extends Node {
  private txtObjects: Txt[] = [];

  public constructor({ children, ...props }: TxtProps) {
    super(props);
    if (!children) {
      throw new Error("Text is required");
    }

    const spacing = props.letterSpacing?.valueOf() as number || 50;

    const txtObjects = children.toString().split("").map((letter, index) => {
      const txt = (
        <Txt
          {...props}
          x={index * spacing}
          y={80}
          opacity={0}
        >
          {letter}
        </Txt>
      ) as Txt;
      return txt;
    });

    const container = <Rect />;
    this.add(container);
    for (const txt of txtObjects) {
      container.add(txt);
    }

    this.txtObjects = txtObjects;
  }

  * flyIn(duration: number, delay: number): ThreadGenerator {
    const generators: ThreadGenerator[] = [];
    // const numLetters = this.txtObjects.length;

    for (const txt of this.txtObjects) {
      generators.push(all(
        txt.opacity(1, duration),
        txt.position.y(0, duration),
      ));
    }

    yield* sequence(delay, ...generators);
  }
}
