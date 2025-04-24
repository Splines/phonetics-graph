import { Node } from "@motion-canvas/2d";

Node.prototype.shiftX = function (deltaX, duration) {
  return this.x(this.x() + deltaX, duration);
};

Node.prototype.shiftY = function (deltaY, duration) {
  return this.y(this.y() + deltaY, duration);
};
