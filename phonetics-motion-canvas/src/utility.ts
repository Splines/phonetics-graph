import { Line, Node } from "@motion-canvas/2d";
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

/**
 * Moves the geometry of a line to its center of mass. It should make no visual
 * difference.
 *
 * The problem is that we specified the line points in such a way that the
 * line is not centered around its center of mass, therefore complicating
 * subsequent animations like rotations.
 *
 * This function adjusts the line's geometry so that it is centered around
 * its center of mass. It does so by calculating the center of mass
 * of the line, then the relative position of the line's points to that center,
 * applying those relative positions to the line's geometry such that the points
 * are centered around the origin, and finally moving the whole line to the
 * center of mass.
 */
export function moveLineGeometryToCenter(line: Line): void {
  const center = centerOfMass(line);

  // Set the line's points to be centered around the origin
  const points = line.points();
  const relativePoints = points.map(point => (point as Vector2).sub(center));
  line.points(relativePoints);

  // Move the line to its center of mass
  line.position(center);
};
