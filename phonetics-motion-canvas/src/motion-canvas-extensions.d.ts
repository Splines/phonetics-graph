import "@motion-canvas/2d";
import { ThreadGenerator } from "@motion-canvas/core";

declare module "@motion-canvas/2d" {
  interface Node {
    /**
     * Shifts the x position of the node by a given delta over a
     * specified duration.
     *
     * @param deltaX - The amount to shift the x position.
     * @param duration - The duration of the shift.
     */
    shiftX(deltaX: number, duration: number): ThreadGenerator;

    /**
     * Shifts the y position of the node by a given delta over a
     * specified duration.
     *
     * @param deltaY - The amount to shift the y position.
     * @param duration - The duration of the shift.
     */
    shiftY(deltaY: number, duration: number): ThreadGenerator;
  }
}
