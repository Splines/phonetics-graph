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
  private container: Rect;

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
  private startX = 0;

  /**
   * Stores the current alignment state for animation.
   */
  private alignment: Alignment[];

  private textReferenceUpMap: Map<number, Txt> = new Map();
  private textReferenceDownMap: Map<number, Txt> = new Map();

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
  constructor(container, word1: string, word2: string, alignmentString: string) {
    this.container = container;

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

    const totalWidth = (alignment.word1.length - 1) * this.WIDTH_TOTAL;
    this.startX = -totalWidth / 2;

    return alignment;
  }

  /**
   * Calculates the x position of a character based on its index.
   *
   * You should call calculateAlignment first as its recomputes
   * the startX value.
   */
  calcPosition(index: number): number {
    return this.startX + index * this.WIDTH_TOTAL;
  }

  /**
   * Creates a text element for a given character.
   * @param char - The character to display.
   * @param position - The x position of the character.
   * @param yShift - The vertical shift for the character.
   * @param opacity - The initial opacity of the character.
   */
  private createTextElement(char: string, position: number, yShift: number, opacity: number): Txt {
    return (
      <Txt
        fontFamily={phoneticFamily}
        fontSize={this.SIZE}
        fill={useScene().variables.get("textFill")}
        opacity={opacity}
        x={position}
        y={yShift}
      >
        {char}
      </Txt>
    );
  }

  generateElements() {
    const elements = [];

    for (let i = 0; i < this.alignment.word1.length; i++) {
      const charUpTxt = this.createTextElement(
        this.alignment.word1[i],
        this.calcPosition(i),
        -this.SHIFT,
        1,
      );

      const charDownTxt = this.createTextElement(
        this.alignment.word2[i],
        this.calcPosition(i),
        0.7 * this.SHIFT,
        1,
      );

      this.textReferenceUpMap.set(i, charUpTxt);
      this.textReferenceDownMap.set(i, charDownTxt);
      elements.push(charUpTxt, charDownTxt);
    }

    return elements;
  }

  /**
   * Finds a mapping between the indices of chars such that the distance
   * between each mapping is minimized.
   */
  mapToNewAlignment(oldWord, newWord) {
    // console.log(`oldWord: ${oldWord}, newWord: ${newWord}`);
    const map = new Map();

    let oldIndex = 0;
    let newIndex = 0;

    while (oldIndex < oldWord.length && newIndex < newWord.length) {
      // console.log(`oldIndex: ${oldIndex}, newIndex: ${newIndex}`);
      const oldChar = oldWord[oldIndex];
      const newChar = newWord[newIndex];

      if (oldChar === newChar) {
        if (oldChar !== "–" && newChar !== "–") {
          // console.log(`Mapping ${oldIndex} -> ${newIndex}`);
          map.set(oldIndex, newIndex);
        }
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

  /**
   * Animates the elements to a specified state.
   *
   * - Letters will be shifted to the new position (via `tween`).
   * - Gaps appear/disappear as needed (via `spawn`).
   */
  * animateToState(newAlignmentString: string, duration: number): ThreadGenerator {
    const newAlignment = this.calculateAlignment(newAlignmentString);
    const generators = [];

    const alignmentAnim = (oldWord, newWord, textRefsMap, yShift) => {
      const map = this.mapToNewAlignment(oldWord, newWord);

      for (let i = 0; i < oldWord.length; i++) {
        const current = textRefsMap.get(i);

        if (map.has(i)) {
          // shift to new position
          const newIndex = map.get(i);
          textRefsMap.set(newIndex, textRefsMap.get(i));
          const newPos = this.calcPosition(newIndex);
          generators.push(current.position.x(newPos, duration));
        } else {
          // remove old gaps
          generators.push(current.opacity(0, 0.8 * duration).do(() => current.remove()));
          textRefsMap.delete(i);
        }
      }

      // show new gaps
      for (let i = 0; i < newWord.length; i++) {
        if (newWord[i] === "–") {
          const charTxt = this.createTextElement(newWord[i], this.calcPosition(i), yShift, 0);
          textRefsMap.set(i, charTxt);
          this.container().add(charTxt);
          generators.push(charTxt.opacity(1, 0.8 * duration));
        }
      }
    };

    alignmentAnim(
      this.alignment.word1, newAlignment.word1,
      this.textReferenceUpMap, -this.SHIFT);
    alignmentAnim(
      this.alignment.word2, newAlignment.word2,
      this.textReferenceDownMap, 0.7 * this.SHIFT);

    this.alignment = newAlignment;
    yield* all(...generators);
  }
}

export default makeScene2D(function* (view) {
  const textFill = useScene().variables.get("textFill");

  const container = createRef<Rect>();
  const alignState = new AlignState(container, "pɥisɑ̃s", "nɥɑ̃s", "-.-.:.-");

  view.add(
    <Rect ref={container}>
      {alignState.generateElements()}
    </Rect>,
  );

  yield* waitFor(0.5);
  yield* alignState.animateToState("..-.:--", 1.2);
  yield* waitFor(0.5);
  yield* alignState.animateToState("....--", 1.2);
  yield* waitFor(0.5);
});
