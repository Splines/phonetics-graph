import { makeScene2D, Txt, Layout, Rect } from "@motion-canvas/2d";
import { useScene } from "@motion-canvas/core/lib/utils";

class AlignState {
  private SIZE = 70;

  constructor(word1: string, word2: string, alignment: string) {
    const segmenter = new Intl.Segmenter("en", { granularity: "grapheme" });
    this.word1 = Array.from(segmenter.segment(word1), segment => segment.segment);
    this.word2 = Array.from(segmenter.segment(word2), segment => segment.segment);
    this.alignment = alignment;
  }

  generateElements() {
    const elements = [];
    const phoneticFamily = "Charis";
    const textFill = useScene().variables.get("textFill");

    for (let i = 0; i < this.word1.length; i++) {
      console.log(this.word1[i]);
      elements.push(
        <Txt
          fontFamily={phoneticFamily}
          fontSize={this.SIZE}
          fill={textFill}
          x={-200 + i * 100}
          y={-50}
        >
          {this.word1[i]}
        </Txt>,
      );
    }

    for (let j = 0; j < this.word2.length; j++) {
      elements.push(
        <Txt
          fontFamily={phoneticFamily}
          fontSize={this.SIZE}
          fill={textFill}
          x={-200 + j * 100}
          y={50}
        >
          {this.word2[j]}
        </Txt>,
      );
    }

    return elements;
  }
}

export default makeScene2D(function* (view) {
  const textFill = useScene().variables.get("textFill");

  const elements = new AlignState("pɥisɑ̃s", "nɥɑ̃s").generateElements();

  view.add(
    <Rect>
      {elements}
    </Rect>,
  );
});
