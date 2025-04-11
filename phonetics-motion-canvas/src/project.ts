import { makeProject } from "@motion-canvas/core";

import test from "./scenes/test?scene";

import "./global.css";

export default makeProject({
  scenes: [test],
  variables: {
    textFill: "white",
  },
});
