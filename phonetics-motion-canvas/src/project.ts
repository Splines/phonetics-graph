import { makeProject } from "@motion-canvas/core";

import alignment from "./scenes/alignment?scene";
import scoring from "./scenes/scoring?scene";

import "./global.css";

export default makeProject({
  scenes: [alignment, scoring],
  variables: {
    textFill: "white",
  },
});
