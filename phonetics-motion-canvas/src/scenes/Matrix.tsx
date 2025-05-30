import { Latex, Layout, Line, Node, Rect, Txt } from "@motion-canvas/2d";
import {
  all, chain, createRef,
  Reference,
  sequence,
  ThreadGenerator,
  Vector2,
  waitFor,
} from "@motion-canvas/core";
import {
  HIGHLIGHT_COLOR,
  HIGHLIGHT_COLOR_2,
  PHONETIC_FAMILY,
  TEXT_FILL,
  TEXT_FILL_DARK,
} from "./globals";

const segmenter = new Intl.Segmenter("en", { granularity: "grapheme" });

export class Matrix {
  public container: Reference<Node>;
  public layout: Reference<Layout>;
  public rects: Rect[] = [];
  public textRects: Rect[] = [];
  public emptyRects: Rect[] = [];

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
          const emptyRect = <Rect width={130} height={130} /> as Rect;
          this.emptyRects.push(emptyRect);
          this.layout().add(emptyRect);
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
    const color = isVariant ? HIGHLIGHT_COLOR_2 : HIGHLIGHT_COLOR;
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
        stroke={isVariant ? HIGHLIGHT_COLOR_2 : HIGHLIGHT_COLOR}
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
        ...sourceRect.children().map((child) => {
          return child instanceof Latex ? child.fill(TEXT_FILL, duration) : null;
        }),
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

  /**
   * Steps from the source cell to the target cell and keeps the arrow visible.
   *
   * Note that this is used for the enlarged matrix grid (with lots of padding),
   * so you should probably use a different offset value than the default if not
   * working in that context.
   */
  public stepAndArrowStay(iSource: number, jSource: number,
    iTarget: number, jTarget: number, duration: number, offset = 45,
    endArrow = true, color = HIGHLIGHT_COLOR): [ThreadGenerator, Line] {
    const sourceRect = this.getRectAt(iSource, jSource);
    const targetRect = this.getRectAt(iTarget, jTarget);

    const arrowStart = new Vector2(
      sourceRect.x() + (jSource !== jTarget ? sourceRect.width() / 2 + offset : 0),
      sourceRect.y() + (iSource !== iTarget ? sourceRect.height() / 2 + offset : 0),
    );
    const arrowEnd = new Vector2(
      targetRect.x() - (jSource !== jTarget ? targetRect.width() / 2 + offset : 0),
      targetRect.y() - (iSource !== iTarget ? targetRect.height() / 2 + offset : 0),
    );
    const arrow = (
      <Line
        points={[
          arrowStart,
          arrowEnd,
        ]}
        lineWidth={18}
        stroke={color}
        endArrow={endArrow}
        lineCap="round"
        opacity={0}
        end={0}
      />
    ) as Line;
    this.container().add(arrow);

    return [all(
      arrow.opacity(1, 0.2 * duration),
      arrow.end(1, duration),
    ), arrow];
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
        stroke={HIGHLIGHT_COLOR}
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

          this.word1Texts[i - 1].fill(HIGHLIGHT_COLOR, duration),
          this.word2Texts[j - 1].fill(HIGHLIGHT_COLOR, duration),
        ),
      ),
    );
  }

  /**
   * Highlights the alignment path in the matrix.
   *
   * - `.` = bottom-right diagonal
   * - `:` = right
   * - `-` = down
   */
  public highlightAlignmentPath(alignmentString: string, duration: number): ThreadGenerator[] {
    const path: number[][] = [[0, 0]];
    let row = 0;
    let col = 0;

    for (const char of alignmentString) {
      if (char === ".") {
        path.push([++row, ++col]);
      } else if (char === ":") {
        path.push([row, ++col]);
      } else if (char === "-") {
        path.push([++row, col]);
      } else {
        throw new Error(`Invalid character in alignment string: ${char}`);
      }
    }

    return path.map((p) => {
      const rect = this.getRectAt(p[0], p[1]);

      const durationShort = duration * 0.6;

      return sequence(durationShort,
        this.highlight(p[0], p[1], durationShort),
        all(
          rect.fill(null, durationShort),
          rect.stroke(HIGHLIGHT_COLOR, durationShort),
          rect.lineWidth(9, durationShort),
        ),
      );
    });
  }

  public getAllRects(): Rect[] {
    return this.rects.concat(this.textRects).concat(this.emptyRects);
  }

  public writeTextAt(row: number, col: number, text: string, duration: number,
    withHighlight = true, fontSize = 61): ThreadGenerator {
    const rect = this.getRectAt(row, col);
    const txt = (
      <Latex
        tex={text}
        fontSize={fontSize}
        fill={withHighlight ? TEXT_FILL_DARK : TEXT_FILL}
        opacity={0}
      >
      </Latex>
    ) as Latex;

    rect.justifyContent("center");
    rect.alignItems("center");
    rect.add(txt);

    if (withHighlight) {
      return all(
        this.highlightRect(rect, duration),
        txt.opacity(1, duration),
      );
    }
    return all(
      txt.opacity(1, duration),
      rect.stroke(HIGHLIGHT_COLOR, duration),
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
    const rect = (
      <Rect
        width={130}
        height={130}
        justifyContent="center"
        alignItems="center"
      >
        <Txt
          fontFamily={PHONETIC_FAMILY}
          fontSize={this.FONT_SIZE}
          fill={TEXT_FILL}
          opacity={0}
        >
          {char}
        </Txt>
      </Rect>
    ) as Rect;

    this.textRects.push(rect);
    return rect;
  }
}
