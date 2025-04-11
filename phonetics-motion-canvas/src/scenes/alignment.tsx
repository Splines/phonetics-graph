import { is, Line, makeScene2D, Node, Rect, Txt } from "@motion-canvas/2d";
import {
  all, chain, createRef, Reference, sequence, ThreadGenerator,
  useScene, waitFor,
} from "@motion-canvas/core";
import { LetterTxt } from "./LetterTxt";

const TEXT_FONT = "Vermiglione";
const PHONETIC_FAMILY = "Charis";

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
  private startX = 0;

  private container: Reference<Node>;
  private alignment: Alignment;
  private textReferenceUpMap: Map<number, Txt> = new Map();
  private textReferenceDownMap: Map<number, Txt> = new Map();

  private word1: string[];
  private word2: string[];

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
  constructor(container: Reference<Node>, word1: string, word2: string, alignmentString: string) {
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
  calculateAlignment(alignmentString: string): Alignment {
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
        fontFamily={PHONETIC_FAMILY}
        fontSize={this.SIZE}
        fill={useScene().variables.get("textFill", "white")}
        opacity={opacity}
        x={position}
        y={yShift}
      >
        {char}
      </Txt>
    ) as Txt;
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
  mapToNewAlignment(oldWord: string[], newWord: string[]) {
    const map = new Map();

    let oldIndex = 0;
    let newIndex = 0;

    // mapping: to new index -> from old index
    const gapShiftedFrom = new Map();

    while (oldIndex < oldWord.length && newIndex < newWord.length) {
      const oldChar = oldWord[oldIndex];
      const newChar = newWord[newIndex];

      if (oldChar === newChar) {
        map.set(oldIndex, newIndex);
        if (oldChar === "–") {
          gapShiftedFrom.set(newIndex, oldIndex);
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

    return [map, gapShiftedFrom];
  }

  /**
   * Animates the elements to a specified state.
   *
   * - Letters will be shifted to the new position (via `tween`).
   * - Gaps appear/disappear as needed (via `spawn`).
   */
  * animateToState(newAlignmentString: string, duration: number): ThreadGenerator {
    const newAlignment = this.calculateAlignment(newAlignmentString);
    const generators: ThreadGenerator[] = [];

    const alignmentAnim = (oldWord: string[], newWord: string[],
      textRefsMap: Map<number, Txt>, yShift: number) => {
      const newTextRefsMap = new Map(textRefsMap);

      const [map, gapShiftedFrom] = this.mapToNewAlignment(oldWord, newWord);

      for (let i = 0; i < oldWord.length; i++) {
        const current = textRefsMap.get(i);

        if (map.has(i)) {
          // shift to new position
          const newIndex = map.get(i);
          newTextRefsMap.set(newIndex, current);
          const newPos = this.calcPosition(newIndex);
          generators.push(current.position.x(newPos, duration));
        } else {
          // remove old gaps
          generators.push(current.opacity(0, 0.8 * duration).do(() => current.remove()));
        }
      }

      // show new gaps
      for (let i = 0; i < newWord.length; i++) {
        if (newWord[i] === "–" && !gapShiftedFrom.has(i)) {
          const charTxt = this.createTextElement(newWord[i], this.calcPosition(i), yShift, 0);
          newTextRefsMap.set(i, charTxt);
          this.container().add(charTxt);
          generators.push(charTxt.opacity(1, 0.8 * duration));
        }
      }

      // replace the old map
      for (const [key, value] of newTextRefsMap) {
        textRefsMap.set(key, value);
      }
    };

    alignmentAnim(this.alignment.word1, newAlignment.word1,
      this.textReferenceUpMap, -this.SHIFT);
    alignmentAnim(this.alignment.word2, newAlignment.word2,
      this.textReferenceDownMap, 0.7 * this.SHIFT);

    this.alignment = newAlignment;
    yield* all(...generators);
  }
}

/**
 * Generates all possible alignment strings for a given minimum word length.
 * Alignments strings can have varying lengths up to the both word lengths
 * added up.
 *
 * - There must be at most `minWordLength` dots (".") in the string.
 * - All remaining chars are "-" or ":" (arbitrary order)
 *
 * It must be guaranteed for the final string that:
 * - num(".") + num("-") = word1Length
 * - num(".") + num(":") = word2Length
 *
 * In an extreme case, we can have a string like
 * ::::------ (word1Length = 6, word2Length = 4)
 */
function generateAllPossibleAlignmentStrings(word1Length: number, word2Length: number): string[] {
  const results: string[] = [];

  function backtrack(current: string, dots: number, dashes: number, colons: number) {
    if (dots + dashes === word1Length && dots + colons === word2Length) {
      results.push(current);
      return;
    }

    if (dots < Math.min(word1Length, word2Length)) {
      backtrack(current + ".", dots + 1, dashes, colons);
    }

    if (dashes < word1Length) {
      backtrack(current + "-", dots, dashes + 1, colons);
    }

    if (colons < word2Length) {
      backtrack(current + ":", dots, dashes, colons + 1);
    }
  }

  backtrack("", 0, 0, 0);
  return results;
}

export default makeScene2D(function* (view) {
  const textFill = useScene().variables.get("textFill", "white");
  const highlightColor = "#FFEB6C";
  // const highlightColor2 = "#68D3F7";
  const highlightColor2 = "#C0B8F2";

  const container = createRef<Node>();
  const alignState = new AlignState(container, "pɥisɑ̃s", "nɥɑ̃s", "....--");
  const texts = alignState.generateElements();

  view.add(
    <Rect ref={container}>
      { texts }
    </Rect>,
  );

  yield* waitFor(0.1);

  let alignmentStrings = generateAllPossibleAlignmentStrings(6, 4);

  const riseAround = alignmentStrings.length * 0.999;
  const c = 0.13;

  // for (let i = 0; i < alignmentStrings.length; i++) {
  //   const fall = Math.exp(-0.04 * i);
  //   const exp = Math.exp(c * (i - riseAround));
  //   const rise = exp / (1 + exp);
  //   const stretch = Math.max(fall + rise, 0.006);

  //   yield* alignState.animateToState(alignmentStrings[i], 0.5 * stretch);
  //   yield* waitFor(0.3 * stretch);
  // }

  yield* waitFor(0.2);
  yield* alignState.animateToState("....--", 1);

  // 🎈 Discuss one alignment
  yield* waitFor(0.5);
  yield* alignState.animateToState("..--..", 1.4);

  yield* waitFor(1);

  yield* sequence(0.2, ...[2, 3, 8, 5, 10, 7].map((i) => {
    return texts[i].fill(highlightColor, 0.6);
  }));

  yield* waitFor(1);

  yield* sequence(0.2, ...[0, 1].map((i) => {
    return texts[i].fill(highlightColor2, 0.6);
  }));

  yield* waitFor(1);

  const textsAll = view.findAll(is(Txt));
  yield* sequence(0.2, ...[4, 10, 6, 11].map((i) => {
    return textsAll[i].fill(highlightColor2, 0.6);
  }));

  yield* waitFor(1);

  const rect = createRef<Rect>();
  view.add(
    <Rect
      ref={rect}
      rotation={180}
      lineWidth={5}
      stroke={textFill}
      width={780}
      height={350}
      radius={6}
      closed
      opacity={0.7}
      end={0}
    />,
  );
  yield* all(
    rect().end(1, 0.8),
    all(...textsAll.map(txt => txt.fill(textFill, 0.8))),
  );
  yield* waitFor(1.6);
  yield* rect().opacity(0, 1.2);
  yield* waitFor(0.2);

  // 🎈 Random alignments
  // const tenRandomNumbers = Array.from({ length: 10 },
  //   () => Math.floor(Math.random() * alignmentStrings.length));
  // for (const i of tenRandomNumbers) {
  //   console.log(alignmentStrings[i]);
  // }
  const randomAlignmentStrings = ["-.--:.-:", ".-:-.-:-", ":---:--.:",
    ":-.--::--", "-..-:.-", "....--"];
  const textTransforms = randomAlignmentStrings.map((str) => {
    return chain(
      alignState.animateToState(str, 0.9),
      waitFor(0.3),
    );
  });

  const toMatrixLine = createRef<Line>();
  view.add(
    <Line
      ref={toMatrixLine}
      points={[
        [-150, 0],
        [150, 0],
      ]}
      lineWidth={10}
      stroke={textFill}
      arrowSize={25}
      endArrow
      lineCap="round"
      lineDash={[20, 20]}
      opacity={0}
      end={0}
    />,
  );

  // const pathText = (
  //   <Txt
  //     fill={textFill}
  //     fontFamily={TEXT_FONT}
  //     fontSize={100}
  //     x={600}
  //     letterSpacing={5}
  //   >
  //     Paths in a grid
  //   </Txt>
  // ) as Txt;
  // view.add(pathText);
  const pathText = (
    <LetterTxt
      fill={textFill}
      fontFamily={TEXT_FONT}
      fontSize={100}
      letterSpacing={50}
      x={350}
    >
      Paths in a grid
    </LetterTxt>
  ) as LetterTxt;
  view.add(pathText);

  yield* all(
    chain(waitFor(1.0), ...textTransforms),
    container().position.x(-800, 5),
    chain(waitFor(2.3), all(toMatrixLine().opacity(1, 0.5), toMatrixLine().end(1, 3))),
    chain(waitFor(4), pathText.flyIn(1, 0.04)),
  );

  yield* waitFor(0.5);
});
