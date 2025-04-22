import { is, Layout, Line, makeScene2D, Node, Rect, Txt } from "@motion-canvas/2d";
import {
  all, chain, createRef, Reference, sequence, ThreadGenerator,
  useScene, Vector2, waitFor,
} from "@motion-canvas/core";
import { Highlight } from "./Highlight";
import { LetterTxt } from "./LetterTxt";

const TEXT_FONT = "Vermiglione";
const PHONETIC_FAMILY = "Charis";

const highlightColor = "#FFEB6C";
// const highlightColor2 = "#68D3F7";
// const highlightColor2 = "#C0B8F2";
const highlightColor2 = "#FF5E7E";

class Alignment {
  word1: string[] = [];
  word2: string[] = [];
}

const segmenter = new Intl.Segmenter("en", { granularity: "grapheme" });

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

  const container = createRef<Node>();
  const alignState = new AlignState(container, "pɥisɑ̃s", "nɥɑ̃s", "....--");
  const texts = alignState.generateElements();
  const alignmentDelta = 100;
  for (const text of texts) {
    text.position.y(text.position.y() + alignmentDelta);
    text.opacity(0);
  }

  view.add(
    <Rect ref={container}>
      { texts }
    </Rect>,
  );

  yield* waitFor(0.1);

  yield* all(
    ...texts.map(txt => txt.position.y(txt.position.y() - alignmentDelta, 0.8)),
    ...texts.map(txt => txt.opacity(1, 0.8)),
  );

  yield* waitFor(1);

  yield* sequence(0.2, ...[0, 1, 2, 3, 4, 5, 6, 7].map((i) => {
    return texts[i].fill(highlightColor, 0.6);
  }));

  yield* waitFor(0.5);

  yield* sequence(0.2, ...[8, 9, 10, 11].map((i) => {
    return texts[i].fill(highlightColor2, 0.6);
  }));

  yield* waitFor(0.5);

  // 🎈 All possible alignments
  let alignmentStrings = generateAllPossibleAlignmentStrings(6, 4);
  const riseAround = alignmentStrings.length * 0.999;
  const c = 0.13;

  // for (let i = 0; i < alignmentStrings.length; i++) {
  //   const fall = Math.exp(-0.04 * i);
  //   const exp = Math.exp(c * (i - riseAround));
  //   const rise = exp / (1 + exp);
  //   const stretch = Math.max(fall + rise, 0.006);

  //   if (i === 0) {
  //     yield* all(
  //       alignState.animateToState(alignmentStrings[i], 0.5 * stretch),
  //       all(...texts.map(txt => txt.fill(textFill, 0.5))),
  //     );
  //   } else {
  //     yield* alignState.animateToState(alignmentStrings[i], 0.5 * stretch);
  //   }
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
    ":-.--::--", "....--"];
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

  const pathText = (
    <LetterTxt
      fill={textFill}
      fontFamily={TEXT_FONT}
      fontSize={100}
      letterSpacing={4}
      x={360}
    >
      Path in a grid
    </LetterTxt>
  ) as LetterTxt;
  view.add(pathText);

  yield* all(
    chain(waitFor(0.8), ...textTransforms),
    container().position.x(-800, 5),
    chain(waitFor(2.3), all(toMatrixLine().opacity(1, 0.5), toMatrixLine().end(1, 3))),
    chain(waitFor(4), pathText.flyIn(1, 0.035)),
  );

  yield* waitFor(0.5);

  // 🎈 Construct the matrix
  const matrixContainerRef = createRef<Node>();
  view.add(
    <Rect
      ref={matrixContainerRef}
      x={520}
      y={-65}
    />,
  );
  const matrix = new Matrix(matrixContainerRef);

  yield* all(
    toMatrixLine().opacity(0, 0.8),
    pathText.opacity(0, 0.8),
    sequence(0.03, ...matrix.rects.map(rect => rect.opacity(1, 1))),
  );

  // Text to new positions
  const animateTextClones = (indices: number[], targetTexts: Txt[]) => {
    const anims: ThreadGenerator[] = [];

    let iTarget = 0;
    for (const i of indices) {
      const original = texts[i];
      const target = targetTexts[iTarget++];

      const clone = original.clone();
      clone.opacity(0);
      view.add(clone);
      clone.absolutePosition(original.absolutePosition());

      anims.push(
        chain(
          all(
            clone.absolutePosition(target.absolutePosition(), 0.7),
            clone.opacity(1, 0.1),
            original.opacity(0, 0.1),
            clone.fontSize(target.fontSize(), 0.7),
          ),
          all(
            // replace clone by target
            clone.opacity(0, 0),
            target.opacity(1, 0),
          ),
        ),
      );
    }

    return anims;
  };

  const gaps = view.findAll(is(Txt)).filter(txt => txt.text() === "–");

  yield* sequence(0.13, ...animateTextClones([0, 2, 4, 6, 8, 10], matrix.word1Texts));
  yield* waitFor(0.3);
  yield* all(
    sequence(0.13, ...animateTextClones([1, 3, 5, 7], matrix.word2Texts)),
    ...gaps.map(txt => txt.opacity(0, 1.2)),
  );

  yield* waitFor(1);

  container().x(container().x() + 300);

  // 🎈 Diagonal steps
  const animateMatrixStep = function* (
    startRow: number,
    startCol: number,
    endRow: number,
    endCol: number,
    highlightIndices: number[],
    duration: number,
    customTexts?: Txt[],
  ) {
    yield* matrix.step(startRow, startCol, endRow, endCol, duration);
    yield* matrix.highlightCoordinates(endRow, endCol, duration);

    let txt = customTexts ? customTexts : texts;
    yield* all(
      ...highlightIndices.map(i => txt[i].opacity(1, duration)),
      ...highlightIndices.map(i => txt[i].fill(highlightColor, duration)),
    );
    yield* waitFor(0.6);
    yield* all(
      ...highlightIndices.map(i => txt[i].fill(textFill, duration)),
    );
  };

  yield* matrix.highlight(0, 0, 0.8);
  yield* waitFor(1.5);
  yield* animateMatrixStep(0, 0, 1, 1, [0, 1], 0.8);
  yield* animateMatrixStep(1, 1, 2, 2, [2, 3], 0.8);
  yield* animateMatrixStep(2, 2, 3, 3, [4, 5], 0.8);
  yield* animateMatrixStep(3, 3, 4, 4, [6, 7], 0.8);
  yield* all(
    matrix.word1Texts[3].fill(textFill, 0.8),
    matrix.word2Texts[3].fill(textFill, 0.8),
  );

  // 🎈 Missing letters in puissance
  const highlight = (
    <Highlight
      width={420}
      height={130}
      x={-300}
      y={-28}
    />
  ) as Highlight;
  view.add(highlight);
  yield* highlight.highlight(1);

  yield* waitFor(1.0);

  yield* sequence(0.5, ...[8, 10].map((i) => {
    return all(
      texts[i].opacity(1, 0.6),
      texts[i].fill(highlightColor2, 0.6),
    );
  }));
  yield* waitFor(0.8);
  yield* sequence(0.5, ...gaps.map((txt) => {
    return all(
      txt.opacity(1, 0.6),
      txt.fill(highlightColor2, 0.6),
    );
  }));
  yield* waitFor(0.2);

  yield* all(
    texts[8].fill(textFill, 0.8),
    texts[10].fill(textFill, 0.8),
    ...gaps.map(txt => txt.fill(textFill, 0.8)),
  );

  yield* animateMatrixStep(4, 4, 5, 4, [], 0.8);
  yield* all(
    texts[8].fill(highlightColor, 0.8),
    gaps[0].fill(highlightColor, 0.8),
  );
  yield* waitFor(0.5);
  yield* all(
    texts[8].fill(textFill, 0.8),
    gaps[0].fill(textFill, 0.8),
  );
  yield* animateMatrixStep(5, 4, 6, 4, [], 0.8);
  yield* all(
    texts[10].fill(highlightColor, 0.8),
    gaps[1].fill(highlightColor, 0.8),
  );
  yield* waitFor(0.5);
  yield* all(
    texts[10].fill(textFill, 1.2),
    gaps[1].fill(textFill, 1.2),
    matrix.word1Texts[5].fill(textFill, 1.2),
    matrix.word2Texts[3].fill(textFill, 1.2),
  );
  yield* waitFor(1);

  // 🎈 Go back some steps
  let duration = 0.5;
  yield* all(
    matrix.getRectAt(6, 4).stroke("white", duration * 1.2),
    matrix.step(6, 4, 5, 4, duration),
    texts[10].opacity(0, duration * 1.5),
    gaps[1].opacity(0, duration * 1.5),
  );
  yield* all(
    matrix.getRectAt(5, 4).stroke("white", duration * 1.2),
    matrix.step(5, 4, 4, 4, duration),
    texts[8].opacity(0, duration * 1.5),
    gaps[0].opacity(0, duration * 1.5),
  );
  yield* all(
    matrix.getRectAt(4, 4).stroke("white", duration * 1.2),
    matrix.step(4, 4, 3, 3, duration),
    texts[6].opacity(0, duration * 1.5),
    texts[7].opacity(0, duration * 1.5),
  );

  yield* waitFor(1);

  const container2 = createRef<Node>();
  const alignment2 = new AlignState(container2, "pɥisɑ̃s", "nɥɑ̃s", "...:---");
  const texts2 = alignment2.generateElements();
  for (const text of texts2) {
    text.position.y(text.position.y() + alignmentDelta);
    text.opacity(0);
  }
  view.add(
    <Rect ref={container2} x={-450} y={-100}>
      { texts2 }
    </Rect>,
  );

  // 🎈 Take horizontal step instead & finish alignment
  duration = 0.4;
  yield* animateMatrixStep(3, 3, 3, 4, [6, 7], 1.0, texts2), // wait
  yield* animateMatrixStep(3, 4, 4, 4, [8, 9], duration, texts2),
  yield* animateMatrixStep(4, 4, 5, 4, [10, 11], duration, texts2),
  yield* animateMatrixStep(5, 4, 6, 4, [12, 13], duration, texts2),
  yield* all(
    matrix.word1Texts[5].fill(textFill, 1.2),
    matrix.word2Texts[3].fill(textFill, 1.2),
  );

  yield* waitFor(1);

  // 🎈 Go back some steps
  duration = 0.5;
  yield* all(
    matrix.getRectAt(6, 4).stroke("white", duration * 1.2),
    matrix.step(6, 4, 5, 4, duration),
    texts2[12].opacity(0, duration * 1.5),
    texts2[13].opacity(0, duration * 1.5),
  );
  yield* all(
    matrix.getRectAt(5, 4).stroke("white", duration * 1.2),
    matrix.step(5, 4, 4, 4, duration),
    texts2[10].opacity(0, duration * 1.5),
    texts2[11].opacity(0, duration * 1.5),
  );
  yield* all(
    matrix.getRectAt(4, 4).stroke("white", duration * 1.2),
    matrix.step(4, 4, 3, 4, duration),
    texts2[8].opacity(0, duration * 1.5),
    texts2[9].opacity(0, duration * 1.5),
  );
  yield* all(
    matrix.getRectAt(3, 4).stroke("white", duration * 1.2),
    matrix.step(3, 4, 3, 3, duration),
    texts2[6].opacity(0, duration * 1.5),
    texts2[7].opacity(0, duration * 1.5),
  );

  yield* waitFor(1);

  // 🎈 Take not-allowed step (bottom-left) & finish alignment
  const container3 = createRef<Node>();
  const alignment3 = new AlignState(container3, "pɥisɑ̃s", "nɥɑ̃ɥɑ̃s", "......");
  const texts3 = alignment3.generateElements();
  for (const text of texts3) {
    text.position.y(text.position.y() + alignmentDelta);
    text.opacity(0);
  }
  view.add(
    <Rect ref={container3} x={-500} y={-100}>
      { texts3 }
    </Rect>,
  );

  duration = 0.8;
  yield* animateMatrixStep(3, 3, 4, 2, [6, 7], duration, texts3);
  yield* waitFor(1);
  yield* animateMatrixStep(4, 2, 5, 3, [8, 9], duration, texts3);
  yield* animateMatrixStep(5, 3, 6, 4, [10, 11], duration, texts3);
  yield* all(
    matrix.word1Texts[5].fill(textFill, 1.2),
    matrix.word2Texts[3].fill(textFill, 1.2),
  );

  yield* waitFor(1);

  const highlightWeird = (
    <Highlight
      width={600}
      height={126}
      x={-252}
      y={30}
    />
  ) as Highlight;
  view.add(highlightWeird);
  yield* highlightWeird.highlight(1.2);

  yield* waitFor(1);

  // 🎈 Highlight first bad move
  duration = 1.0;
  const startRect = matrix.getRectAt(3, 3);
  yield* all(
    startRect.lineWidth(10, duration),
    startRect.stroke(highlightColor2, duration),
    startRect.fill(highlightColor2, duration),
    matrix.step(3, 3, 4, 2, duration, true),
  );

  yield* waitFor(1);

  // 🎈 Reset all matrix fields except (3,3) and (4,2) and move matrix to center
  duration = 2.0;

  let resetMatrix: ThreadGenerator[] = [];
  for (let row = 0; row < matrix.numRows; row++) {
    for (let col = 0; col < matrix.numCols; col++) {
      if ((row === 3 && col === 3) || (row === 4 && col === 2)) {
        continue;
      }
      const rect = matrix.getRectAt(row, col);
      resetMatrix.push(
        all(
          rect.fill(null, duration),
          rect.stroke(matrix.BASE_COLOR, duration),
        ),
      );
    }
  }
  yield* all(
    matrix.container().x(0, 2.5),
    ...texts.map(txt => txt.opacity(0, duration)),
    ...gaps.map(txt => txt.opacity(0, duration)),
    ...texts2.map(txt => txt.opacity(0, duration)),
    ...texts3.map(txt => txt.opacity(0, duration)),
    ...resetMatrix,
  );

  yield* waitFor(1);

  // 🎈 Highlight all bad & good moves
  const badMoveEnds = [[3, 2], [2, 2], [2, 3], [2, 4]];
  yield* chain(
    ...badMoveEnds.map(([row, col]) => {
      return matrix.step(3, 3, row, col, 0.8, true);
    }),
  );
  yield* matrix.getRectAt(3, 3).stroke(matrix.BASE_COLOR, 0.8);

  yield* waitFor(1);

  const goodMoveEnds = [[3, 4], [4, 4], [4, 3]];
  yield* chain(
    ...goodMoveEnds.map(([row, col]) => {
      return matrix.step(3, 3, row, col, 0.8);
    }),
  );

  yield* waitFor(5);
});

class Matrix {
  public container: Reference<Node>;
  private layout: Reference<Layout>;
  public rects: Rect[] = [];

  public numRows = 6 + 1; // puissance
  public numCols = 4 + 1; // nuance

  public BASE_COLOR = "#EDEDED";
  private FONT_SIZE = 80;
  private COORD_COLOR = "#54B0CF";

  private word1 = Array.from(segmenter.segment("pɥisɑ̃s"), segment => segment.segment);
  private word2 = Array.from(segmenter.segment("nɥɑ̃s"), segment => segment.segment);
  public word1Texts: Txt[] = [];
  public word2Texts: Txt[] = [];

  constructor(container: Reference<Node>) {
    this.container = container;
    const layout = createRef<Layout>();
    this.layout = layout;
    container().add(<Layout ref={layout} layout gap={20} width={900} wrap="wrap" />);

    for (const char of [" ", " ", ...this.word2]) {
      const txt = this.createTextInRect(char);
      if (char !== " ") {
        this.word2Texts.push(txt.childAs(0));
      }
      this.layout().add(txt);
    }

    for (let i = 1; i < this.numRows + 1; i++) {
      for (let j = 0; j < this.numCols + 1; j++) {
        if (i >= 2 && j === 0) {
          const txt = this.createTextInRect(this.word1[i - 2]);
          this.word1Texts.push(txt.childAs(0));
          this.layout().add(txt);
          continue;
        }

        if (j === 0) {
          this.layout().add(<Rect width={130} height={130} />);
          continue;
        }

        const rect = this.getRect();
        this.rects.push(rect);
        this.layout().add(rect);
      }
    }
  }

  private calcIndex(i: number, j: number): number {
    return i * this.numCols + j;
  }

  public getRectAt(i: number, j: number): Rect {
    const index = this.calcIndex(i, j);
    return this.rects[index];
  }

  public highlight(i: number, j: number, duration: number): ThreadGenerator {
    const rect = this.getRectAt(i, j);
    return this.highlightRect(rect, duration);
  }

  private highlightRect(rect: Rect, duration: number,
    isVariant: boolean = false): ThreadGenerator {
    const color = isVariant ? highlightColor2 : highlightColor;
    const strokeAnim = isVariant ? rect.lineWidth(10, duration) : null;

    return all(
      rect.stroke(color, duration),
      rect.fill(color, duration),
      strokeAnim,
    );
  }

  public step(iSource: number, jSource: number,
    iTarget: number, jTarget: number, duration: number,
    isVariant: boolean = false): ThreadGenerator {
    const sourceRect = this.getRectAt(iSource, jSource);
    const targetRect = this.getRectAt(iTarget, jTarget);

    let xOffset = 0;
    let yOffset = 0;
    // diagonal step
    if (iSource === iTarget - 1 && jSource === jTarget - 1) {
      xOffset = sourceRect.width() / 6;
      yOffset = sourceRect.height() / 6;
    }

    const arrowStart = new Vector2(
      sourceRect.x() + xOffset,
      sourceRect.y() + yOffset,
    );
    const arrowEnd = new Vector2(
      targetRect.x() - xOffset,
      targetRect.y() - yOffset,
    );

    const arrow = (
      <Line
        points={[
          arrowStart,
          arrowEnd,
        ]}
        lineWidth={18}
        stroke={isVariant ? highlightColor2 : highlightColor}
        lineCap="round"
        opacity={0}
        end={0}
      />
    ) as Line;
    this.container().add(arrow);

    const words1Anims = isVariant ? [] : this.word1Texts.map(txt => txt.fill("white", 0.8));
    const words2Anims = this.word2Texts.map(txt => txt.fill("white", 0.8));

    return all(
      all(
        ...words1Anims,
        ...words2Anims,
        sourceRect.fill(null, duration),
        arrow.opacity(1, duration),
        arrow.end(1, duration),
      ),
      chain(
        waitFor(0.5 * duration),
        all(
          arrow.start(1, duration),
          chain(waitFor(0.3 * duration), arrow.opacity(0, duration)),
          this.highlightRect(targetRect, duration, isVariant),
        ),
      ),
    );
  }

  public highlightCoordinates(i: number, j: number, duration: number): ThreadGenerator {
    const rect = this.getRectAt(i, j);

    const arrowHorizontal = (
      <Line
        points={[
          new Vector2(rect.position().x, rect.position().y),
          new Vector2(rect.position().x - (j + 1) * (rect.width() + 10), rect.position().y),
        ]}
        lineWidth={12}
        stroke={highlightColor}
        lineCap="round"
        opacity={0}
        lineDash={[10, 30]}
        end={0}
      />
    ) as Line;

    const arrowVertical = arrowHorizontal.clone() as Line;
    arrowVertical.points([
      new Vector2(rect.position().x, rect.position().y),
      new Vector2(rect.position().x, rect.position().y - (i + 1) * (rect.height() + 10)),
    ]);

    this.container().add(arrowHorizontal);
    this.container().add(arrowVertical);

    return all(
      arrowHorizontal.opacity(1, 0.6 * duration),
      arrowHorizontal.end(1, duration),
      arrowVertical.opacity(1, 0.6 * duration),
      arrowVertical.end(1, duration),
      chain(
        waitFor(1.2 * duration),
        all(
          arrowHorizontal.start(1, duration),
          arrowHorizontal.opacity(0, duration),
          arrowVertical.start(1, duration),
          arrowVertical.opacity(0, duration),

          this.word1Texts[i - 1].fill(highlightColor, duration),
          this.word2Texts[j - 1].fill(highlightColor, duration),
        ),
      ),
    );
  }

  private getRect(): Rect {
    return new Rect({
      width: 130,
      height: 130,
      stroke: this.BASE_COLOR,
      lineWidth: 6,
      radius: 8,
      opacity: 0,
    });
  }

  private createTextInRect(char: string): Rect {
    return (
      <Rect
        width={130}
        height={130}
        justifyContent="center"
        alignItems="center"
      >
        <Txt
          fontFamily={PHONETIC_FAMILY}
          fontSize={this.FONT_SIZE}
          fill={useScene().variables.get("textFill", "white")}
          opacity={0}
        >
          {char}
        </Txt>
      </Rect>
    ) as Rect;
  }
}
