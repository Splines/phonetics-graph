import { Node } from "@motion-canvas/2d";
import { ThreadGenerator, Vector2 } from "@motion-canvas/core";

/**
 * Returns a vector that can be used to position the center of mass of a node.
 */
export function positionAtCenterOfMass(
  node: Node, x: number, y: number, duration = 0.0): ThreadGenerator {
  const center = centerOfMass(node);
  const pos = new Vector2(x - center.x, y - center.y);
  return node.position(pos, duration);
}

function centerOfMass(node: Node): Vector2 {
  const bounds = node.cacheBBox();
  return new Vector2(
    (bounds.left + bounds.right) / 2,
    (bounds.top + bounds.bottom) / 2,
  );
}

export function centerOfMassAbs(node: Node): Vector2 {
  const center = centerOfMass(node);
  return node.absolutePosition().add(center);
}
