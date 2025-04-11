import { makeProject } from "@motion-canvas/core";

import alignment from "./scenes/alignment?scene";
import test from "./scenes/test?scene";

import "./global.css";

export default makeProject({
  scenes: [test, alignment],
  variables: {
    textFill: "white",
  },
});
