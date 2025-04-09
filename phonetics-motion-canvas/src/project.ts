import {makeProject} from '@motion-canvas/core';

import alignment from './scenes/alignment?scene';

import './global.css';

export default makeProject({
    scenes: [alignment],
  variables: {
    textFill: "white"
  }
});
