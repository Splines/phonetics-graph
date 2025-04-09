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
   * Constructs an instance of `AlignState`.
   *
   * @param word1 - The first word to be aligned.
   * @param word2 - The second word to be aligned.
   * @param alignment - A string representing the alignment pattern.
   *                    It can contain the following characters:
   *                    - `"-"`: gap in the second word.
   *                    - `":"`: gap in the first word.
   *                    - `"."`: match between characters in both words.
   *
   * @throws Throws an error if an invalid alignment character is encountered.
   */
  constructor(word1: string, word2: string, alignment: string) {
    const segmenter = new Intl.Segmenter("en", { granularity: "grapheme" });
    this.word1 = Array.from(segmenter.segment(word1), segment => segment.segment);
    this.word2 = Array.from(segmenter.segment(word2), segment => segment.segment);
    this.alignment = alignment;
  }

  /**
   * Calculates the alignment between the two words
   * based on the alignment string.
   *
   * @returns An array of alignment objects, where each object contains:
   *          - `char1`: The character from the first word or `"-"` for a gap.
   *          - `char2`: The character from the second word or `"-"` for a gap.
   *          - `index`: The index of the alignment in the alignment string.
   *
   * @throws Throws an error if an invalid alignment character is encountered.
   */
  calculateAlignment() {
    const result = [];
    let i = 0, j = 0;

    for (let k = 0; k < this.alignment.length; k++) {
      const alignChar = this.alignment[k];

      if (alignChar === "-") {
        result.push({ char1: this.word1[i++], char2: "–", index: k });
      } else if (alignChar === ":") {
        result.push({ char1: "–", char2: this.word2[j++], index: k });
      } else if (alignChar === ".") {
        result.push({ char1: this.word1[i++], char2: this.word2[j++], index: k });
      } else {
        throw new Error(`Invalid alignment character: ${alignChar}`);
      }
    }

    return result;
  }

  generateElements() {
    const elements = [];
    const textFill = useScene().variables.get("textFill");
    const alignmentData = this.calculateAlignment();
    const widthTotal = 100;

    const totalWidth = (alignmentData.length - 1) * widthTotal;
    const startX = -totalWidth / 2;

    for (const { char1, char2, index } of alignmentData) {
      elements.push(
        <Txt
          fontFamily={phoneticFamily}
          fontSize={this.SIZE}
          fill={textFill}
          x={startX + index * widthTotal}
          y={-this.SHIFT}
        >
          {char1}
        </Txt>,
      );

      elements.push(
        <Txt
          fontFamily={phoneticFamily}
          fontSize={this.SIZE}
          fill={textFill}
          x={startX + index * widthTotal}
          y={0.85 * this.SHIFT}
        >
          {char2}
        </Txt>,
      );
    }

    return elements;
  }

  /**
   * Animates the elements to a specified state.
   *
   * - Letters will be shifted to the new position (via `tween`).
   * - Gaps appear/disappear as needed (via `spawn`).
   */
  * animateToState(state: string, duration: number): ThreadGenerator {
    yield* waitFor(1);
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

  yield* alignState.animateToState("....--");
});
