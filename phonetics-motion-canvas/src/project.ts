import { makeProject } from "@motion-canvas/core";

// import alignment from "./scenes/alignment?scene";
// import bug from "./scenes/bug?scene";
// import scoring from "./scenes/scoring?scene";
import code from "./scenes/code?scene";

import "./global.css";

export default makeProject({
  scenes: [code],
});
