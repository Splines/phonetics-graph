import { Layout, Line, Node, Rect, Txt } from "@motion-canvas/2d";
import {
  all, chain, createRef,
  Reference,
  ThreadGenerator,
  useScene,
  Vector2,
  waitFor,
} from "@motion-canvas/core";
import { HIGHLIGHT_COLOR, HIGHLIGHT_COLOR_2, PHONETIC_FAMILY } from "./globals";

const segmenter = new Intl.Segmenter("en", { granularity: "grapheme" });

export class Matrix {
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
