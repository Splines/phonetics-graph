import { makeScene2D, Txt, Rect } from "@motion-canvas/2d";
import {
  createEffect,
  createRef,
  createSignal,
  spawn,
  waitFor,
  useScene,
  tween,
  all,
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
   * The total width of the text elements in the alignment.
   */
  private WIDTH_TOTAL = 100;

  /**
   * The position of the leftmost text element in the alignment.
   */
  private START_X = 0;

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

    const totalWidth = (this.alignment.word1.length - 1) * this.WIDTH_TOTAL;
    this.START_X = -totalWidth / 2;
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

  calcPosition(index: number): number {
    return this.START_X + index * this.WIDTH_TOTAL;
  }

  generateElements() {
    const elements = [];
    const textFill = useScene().variables.get("textFill");

    for (let i = 0; i < this.alignment.word1.length; i++) {
      const charUpTxt = (
        <Txt
          fontFamily={phoneticFamily}
          fontSize={this.SIZE}
          fill={textFill}
          x={this.calcPosition(i)}
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
          x={this.calcPosition(i)}
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
    const newAlignment = this.calculateAlignment(newAlignmentString);
    const generators = [];

    /**
     * Finds a mapping between the indices of chars such that the distance
     * between each mapping is minimized.
     */
    function mapToNewAlignment(oldWord, newWord) {
      const map = new Map();

      let oldIndex = 0;
      let newIndex = 0;

      while (oldIndex < oldWord.length && newIndex < newWord.length) {
        const oldChar = oldWord[oldIndex];
        const newChar = newWord[newIndex];

        if (oldChar === newChar) {
          if (oldChar === "–" || newChar === "–") {
            // to account for "-" at the end
            break;
          }
          map.set(oldIndex, newIndex);
          oldIndex++;
          newIndex++;
        } else {
          if (newChar === "–") {
            newIndex++;
          }
          if (oldChar === "–") {
            oldIndex++;
          }
        }
      }

      return map;
    }

    const mapUp = mapToNewAlignment(this.alignment.word1, newAlignment.word1);
    const mapDown = mapToNewAlignment(this.alignment.word2, newAlignment.word2);

    // iterate over every char from left to right
    for (let i = 0; i < this.alignment.word1.length; i++) {
      const current = this.textReferences[i];

      if (current) {
        if (mapUp.has(i)) {
          const newPosUp = this.calcPosition(mapUp.get(i));
          generators.push(current.up.position.x(newPosUp, duration));
        } else {
          generators.push(current.up.opacity(0, duration).do(() => current.up.remove()));
        }

        if (mapDown.has(i)) {
          const newPosDown = this.calcPosition(mapDown.get(i));
          generators.push(current.down.position.x(newPosDown, duration));
        } else {
          generators.push(current.down.opacity(0, duration).do(() => current.down.remove()));
        }
      }
    }

    this.alignment = newAlignment;

    yield* all(...generators);
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

  yield* waitFor(0.5);
  yield* alignState.animateToState("....--", 1);
});
