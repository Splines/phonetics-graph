import { makeScene2D, Txt, Rect } from "@motion-canvas/2d";
import {
  createEffect,
  createRef,
  createSignal,
  spawn,
  waitFor,
  useScene,
  tween,
} from "@motion-canvas/core";

const phoneticFamily = "Charis";

class Alignment {
  word1: string[] = [];
  word2: string[] = [];
}

/**
 * The `AlignState` class is responsible for managing the alignment of two words
 * and generating corresponding visual elements for rendering. It uses grapheme
 * segmentation to handle multi-character graphemes correctly.
 */
class AlignState {
  /**
   * The size of the font used for rendering text elements.
   */
  private SIZE = 90;

  /**
   * The vertical shift applied to the text elements for alignment.
   */
  private SHIFT = 70;

  /**
   * Stores the current alignment state for animation.
   */
  private alignment: Alignment[];

  /**
   * Stores references to the text elements created in `generateElements`.
   */
  private textReferences: { [index: number]: { up: Txt; down: Txt } } = {};

  /**
   * Constructs an instance of `AlignState`.
   *
   * @param word1 - The first word to be aligned (up).
   * @param word2 - The second word to be aligned (down).
   * @param alignment - A string representing the alignment pattern.
   *                    It can contain the following characters:
   *                    - `"-"`: gap in the second word.
   *                    - `":"`: gap in the first word.
   *                    - `"."`: match between characters in both words.
   */
  constructor(word1: string, word2: string, alignmentString: string) {
    const segmenter = new Intl.Segmenter("en", { granularity: "grapheme" });
    this.word1 = Array.from(segmenter.segment(word1), segment => segment.segment);
    this.word2 = Array.from(segmenter.segment(word2), segment => segment.segment);
    this.alignment = this.calculateAlignment(alignmentString);
  }

  /**
   * Calculates the alignment between the two words
   * based on the alignment string.
   */
  calculateAlignment(alignmentString: string): Alignment[] {
    const alignment = new Alignment();
    let i = 0, j = 0;

    for (let k = 0; k < alignmentString.length; k++) {
      const alignChar = alignmentString[k];

      if (alignChar === "-") {
        alignment.word1.push(this.word1[i++]);
        alignment.word2.push("–");
      } else if (alignChar === ":") {
        alignment.word1.push("–");
        alignment.word2.push(this.word2[j++]);
      } else if (alignChar === ".") {
        alignment.word1.push(this.word1[i++]);
        alignment.word2.push(this.word2[j++]);
      } else {
        throw new Error(`Invalid alignment character: ${alignChar}`);
      }
    }

    if (alignment.word1.length !== alignment.word2.length) {
      throw new Error("Alignment lengths do not match");
    }

    return alignment;
  }

  generateElements() {
    const elements = [];
    const textFill = useScene().variables.get("textFill");
    const widthTotal = 100;

    const totalWidth = (this.alignment.word1.length - 1) * widthTotal;
    const startX = -totalWidth / 2;

    for (let i = 0; i < this.alignment.word1.length; i++) {
      const charUpTxt = (
        <Txt
          fontFamily={phoneticFamily}
          fontSize={this.SIZE}
          fill={textFill}
          x={startX + i * widthTotal}
          y={-this.SHIFT}
        >
          {this.alignment.word1[i]}
        </Txt>
      );

      const charDownTxt = (
        <Txt
          fontFamily={phoneticFamily}
          fontSize={this.SIZE}
          fill={textFill}
          x={startX + i * widthTotal}
          y={0.7 * this.SHIFT}
        >
          {this.alignment.word2[i]}
        </Txt>
      );

      this.textReferences[i] = { up: charUpTxt, down: charDownTxt };
      elements.push(charUpTxt, charDownTxt);
    }

    return elements;
  }

  /**
   * Animates the elements to a specified state.
   *
   * - Letters will be shifted to the new position (via `tween`).
   * - Gaps appear/disappear as needed (via `spawn`).
   */
  * animateToState(newAlignmentString: string, duration: number): ThreadGenerator {
    // const newAlignment = this.calculateAlignment(newAlignmentString);

    // for (const { index, _char1, _char2 } of newAlignment) {
    //   const current = this.textReferences[index];

    //   if (current) {
    //     yield* current.char1.position.x(index * 100, duration);
    //     yield* current.char2.position.x(index * 100, duration);
    //   }
    // }

    // this.alignment = newAlignment;
  }
}

export default makeScene2D(function* (view) {
  const textFill = useScene().variables.get("textFill");

  const alignState = new AlignState("pɥisɑ̃s", "nɥɑ̃s", "-.-.:.-");

  view.add(
    <Rect>
      {alignState.generateElements()}
    </Rect>,
  );

  // yield* alignState.animateToState("....--", 1);
});
